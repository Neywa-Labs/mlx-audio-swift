import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXNN
import MLXRandom
import Tokenizers

private let textEncoderDirectory = "t5gemma-b-b-ul2"
private let textMaxLength = 256
private let secondsMax = 384.0
private let samplesPerLatent = 4096

private let weightPrefixes: [(String, String)] = [
    ("model.model.", "dit."),
    ("conditioner.conditioners.prompt.padding_embedding", "padding_embedding"),
    ("conditioner.conditioners.seconds_total.embedder.embedding.1.", "seconds_proj."),
    ("pretransform.model.bottleneck.running_std", "decoder.running_std"),
    ("pretransform.model.decoder.layers.1.", "decoder.project_in."),
    ("pretransform.model.decoder.layers.3.", "decoder."),
]
private let unusedWeightPrefixes = [
    "pretransform.model.encoder.",
    "pretransform.model.bottleneck.bias",
    "pretransform.model.bottleneck.scaling_factor",
]

public struct StableAudio3GenerationOptions: Sendable {
    public var duration: Double
    public var steps: Int
    public var seed: UInt64
    public var cfgScale: Float
    public var negativePrompt: String?

    public init(duration: Double = 30, steps: Int = 8, seed: UInt64 = 0, cfgScale: Float = 1,
                negativePrompt: String? = nil) {
        self.duration = duration
        self.steps = steps
        self.seed = seed
        self.cfgScale = cfgScale
        self.negativePrompt = negativePrompt
    }
}

func stableAudio3SamplingSchedule(steps: Int) -> MLXArray {
    let logSNRStart = -6.2
    let logSNREnd = 2.0
    let t = MLXArray.linspace(Float(1), Float(0), count: steps + 1)
    let shifted = sigmoid(-(Float(logSNREnd) - t * Float(logSNREnd - logSNRStart)))
    let zeroed = MLX.where(t .<= 0, zeros(like: shifted), shifted)
    return MLX.where(t .>= 1, ones(like: zeroed), zeroed)
}

func stableAudio3GuidedVelocity(x: MLXArray, sigma: MLXArray, conditional: MLXArray, unconditional: MLXArray,
                                cfgScale: Float) -> MLXArray {
    let cond = x - sigma * conditional
    let diff = sigma * (unconditional - conditional)
    let norm = sqrt((cond * cond).sum(axes: [-2, -1], keepDims: true))
    let unit = cond / maximum(norm, MLXArray(Float(1e-8)))
    let orthogonal = diff - (diff * unit).sum(axes: [-2, -1], keepDims: true) * unit
    return (x - (cond + (cfgScale - 1) * orthogonal)) / sigma
}

func stableAudio3PingPong(model: (MLXArray, MLXArray) -> MLXArray, start: MLXArray, sigmas: MLXArray,
                          key startKey: MLXArray) -> MLXArray {
    var x = start
    var key = startKey
    let last = sigmas.dim(0) - 2
    for step in 0 ... last {
        let sigma = sigmas[step]
        let denoised = x - sigma * model(x, sigma.expandedDimensions(axis: 0))
        if step == last {
            x = denoised
        } else {
            let nextSigma = sigmas[step + 1]
            let (nextKey, noiseKey) = MLXRandom.split(key: key)
            key = nextKey
            let noise = MLXRandom.normal(x.shape, dtype: x.dtype, key: noiseKey)
            x = (1 - nextSigma) * denoised + nextSigma * noise
        }
        eval(x)
    }
    return x
}

private struct StableAudio3Configuration: Decodable {
    var sampleRate: Int
    var sampleSize: Int

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case sampleSize = "sample_size"
    }
}

public final class StableAudio3Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public let sampleRate: Int
    let maxDuration: Double
    public let defaultGenerationParameters = GenerateParameters(seed: 0)

    @ModuleInfo var dit: StableAudio3DiT
    @ModuleInfo var decoder: StableAudio3Decoder
    @ModuleInfo(key: "text_encoder") var textEncoder: StableAudio3TextEncoder
    @ModuleInfo(key: "seconds_proj") var secondsProjection: Linear
    @ParameterInfo(key: "padding_embedding") var paddingEmbedding: MLXArray

    var tokenizer: (any Tokenizer)?

    init(sampleRate: Int = 44100, sampleSize: Int = 5_292_032) {
        self.sampleRate = sampleRate
        self.maxDuration = Double(sampleSize / sampleRate)
        _dit.wrappedValue = StableAudio3DiT()
        _decoder.wrappedValue = StableAudio3Decoder()
        _textEncoder.wrappedValue = StableAudio3TextEncoder()
        _secondsProjection.wrappedValue = Linear(StableAudio3Constants.fourierDim, StableAudio3Constants.conditioningDim)
        _paddingEmbedding.wrappedValue = MLXArray.zeros([StableAudio3Constants.conditioningDim])
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasSuffix("inv_freq") || unusedWeightPrefixes.contains(where: key.hasPrefix) { continue }
            var name = key
            for (old, new) in weightPrefixes where key.hasPrefix(old) {
                name = new + key.dropFirst(old.count)
                break
            }
            if name.hasPrefix("dit."), name.hasSuffix(".gamma") {
                name = String(name.dropLast("gamma".count)) + "weight"
            }
            sanitized[name] = value
        }
        return sanitized
    }

    func encodeText(_ text: String) throws -> MLXArray {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Stable Audio 3 tokenizer is not loaded")
        }
        let tokens = Array(tokenizer.encode(text: text, addSpecialTokens: false).prefix(textMaxLength))
        var ids = [Int32](repeating: 0, count: textMaxLength)
        for (index, token) in tokens.enumerated() {
            ids[index] = Int32(token)
        }
        let mask = MLXArray(0 ..< textMaxLength).expandedDimensions(axis: 0) .< tokens.count
        let hidden = textEncoder(MLXArray(ids).reshaped([1, textMaxLength]), attentionMask: mask).asType(.float32)
        let keep = mask.expandedDimensions(axis: -1).asType(.float32)
        return hidden * keep + paddingEmbedding.reshaped([1, 1, -1]) * (1 - keep)
    }

    func secondsEmbedding(_ seconds: Double) -> MLXArray {
        let wholeSeconds = Foundation.ceil(seconds)
        return secondsProjection(stableAudio3FourierFeatures(MLXArray([Float(wholeSeconds / secondsMax)])))
    }

    func latentLength(seconds: Double) -> Int {
        max(1, Int(Foundation.ceil(seconds * Double(sampleRate) / Double(samplesPerLatent))))
    }

    func sample(text: String, options: StableAudio3GenerationOptions) throws -> MLXArray {
        let length = latentLength(seconds: options.duration)
        let seconds = secondsEmbedding(options.duration)
        var crossAttention = concatenated([try encodeText(text), seconds.expandedDimensions(axis: 1)], axis: 1)
        var globalCond = seconds
        let guided = options.cfgScale != 1
        if guided {
            let unconditional: MLXArray
            if let negative = options.negativePrompt, !negative.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                unconditional = concatenated([try encodeText(negative), seconds.expandedDimensions(axis: 1)], axis: 1)
            } else {
                unconditional = zeros(like: crossAttention)
            }
            crossAttention = concatenated([crossAttention, unconditional], axis: 0)
            globalCond = concatenated([seconds, seconds], axis: 0)
        }
        let cond = crossAttention
        let global = globalCond
        let cfgScale = options.cfgScale

        let model: (MLXArray, MLXArray) -> MLXArray = { [dit] x, t in
            guard guided else { return dit(x, t: t, crossAttention: cond, globalCond: global) }
            let velocity = dit(
                concatenated([x, x], axis: 0),
                t: concatenated([t, t], axis: 0),
                crossAttention: cond,
                globalCond: global
            )
            let parts = split(velocity, parts: 2, axis: 0)
            return stableAudio3GuidedVelocity(
                x: x, sigma: t.reshaped([-1, 1, 1]), conditional: parts[0], unconditional: parts[1], cfgScale: cfgScale)
        }

        let (key, noiseKey) = MLXRandom.split(key: MLXRandom.key(options.seed))
        let noise = MLXRandom.normal([1, length, StableAudio3Constants.latentChannels], key: noiseKey)
        return stableAudio3PingPong(
            model: model, start: noise, sigmas: stableAudio3SamplingSchedule(steps: options.steps), key: key)
    }

    public func generate(text: String, options: StableAudio3GenerationOptions = .init()) throws -> MaterializedArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("Stable Audio 3 needs a text prompt")
        }
        guard options.duration > 0, options.duration <= maxDuration else {
            throw AudioGenerationError.invalidInput("Stable Audio 3 duration must be above 0 and at most \(maxDuration) s")
        }
        guard options.steps >= 1 else {
            throw AudioGenerationError.invalidInput("Stable Audio 3 needs at least 1 step")
        }
        let latents = try sample(text: text, options: options)
        let frames = Int((options.duration * Double(sampleRate)).rounded(.toNearestOrEven))
        let audio = decoder.decode(latents)[0, ..<frames]
        let waveform = clip(audio, min: -1, max: 1)
        eval(waveform)
        guard !any(isNaN(waveform)).item(Bool.self) else {
            throw AudioGenerationError.generationFailed("Stable Audio 3 produced non-finite audio")
        }
        return waveform.materialized()
    }

    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        guard refAudio == nil else {
            throw AudioGenerationError.invalidInput("Stable Audio 3 does not take reference audio")
        }
        return try generate(text: text, options: .init(seed: generationParameters.seed ?? 0))
    }

    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        let options = StableAudio3GenerationOptions(seed: generationParameters.seed ?? 0)
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Stable Audio 3 model was released"))
                return
            }
            do {
                let start = Date()
                let audio = try self.generate(text: text, options: options)
                let elapsed = Date().timeIntervalSince(start)
                continuation.yield(.audio(audio))
                continuation.yield(.info(AudioGenerationInfo(
                    promptTokenCount: 0, generationTokenCount: audio.shape[0], prefillTime: 0, generateTime: elapsed,
                    tokensPerSecond: Double(audio.shape[0]) / max(elapsed, 0.001),
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> StableAudio3Model {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: "safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> StableAudio3Model {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(StableAudio3Configuration.self, from: data)
        let model = StableAudio3Model(sampleRate: config.sampleRate, sampleSize: config.sampleSize)

        var weights = sanitize(try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors")))
        let textDirectory = directory.appendingPathComponent(textEncoderDirectory)
        for (key, value) in try MLX.loadArrays(url: textDirectory.appendingPathComponent("model.safetensors"))
            where key.hasPrefix("model.encoder.") {
            weights["text_encoder." + key.dropFirst("model.encoder.".count)] = value
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: textDirectory)
        return model
    }
}
