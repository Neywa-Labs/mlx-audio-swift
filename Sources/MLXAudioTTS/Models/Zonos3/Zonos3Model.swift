import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXNN

/// ZONOS3-Mini synthesis. Serialize calls on each model instance.
public final class Zonos3Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public static let defaultRepositoryID = "mlx-community/ZONOS3-Mini"
    public let config: Zonos3Configuration
    public var sampleRate: Int { 44100 }
    public var voices: [String] { voiceFiles.keys.sorted() }
    public var defaultGenerationParameters: GenerateParameters {
        .init(
            maxTokens: 2000, temperature: 0.75, topP: 0.69, topK: 80,
            repetitionPenalty: 2.4, repetitionContextSize: 96)
    }
    let backbone: Zonos3Backbone
    let directory: URL
    let tokenizer: Zonos3TextTokenizer
    let voiceFiles: [String: String]
    private var codec: Zonos3Codec?
    private var speakerEncoder: Zonos3SpeakerEncoder?

    private init(
        config: Zonos3Configuration, directory: URL, params: Zonos3TensorStore,
        tokenizer: Zonos3TextTokenizer, voices: [String: String]
    ) {
        self.config = config
        self.directory = directory
        backbone = Zonos3Backbone(config.backbone, params: params)
        self.tokenizer = tokenizer
        voiceFiles = voices
    }
    public static func fromPretrained(
        _ repository: String = defaultRepositoryID,
        cache: HubCache = .default, hfToken: String? = nil
    ) async throws -> Zonos3Model {
        guard let id = Repo.ID(rawValue: repository) else {
            throw TTSModelError.invalidRepositoryID(repository)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: id,
            requiredExtension: ".safetensors", additionalMatchingPatterns: ["*.npz"],
            hfToken: hfToken, cache: cache)
        return try await fromModelDirectory(directory)
    }
    public static func fromModelDirectory(_ directory: URL) async throws -> Zonos3Model {
        let config = try Zonos3Configuration.load(directory.appendingPathComponent("config.json"))
        guard ["zonos3", "zonos3_mini"].contains(config.modelType) else {
            throw Zonos3Error.invalidCheckpoint("Expected a native ZONOS3 checkpoint")
        }
        let weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        let params = try Zonos3TensorStore(
            specs: config.tensors, weights: weights, prefix: "backbone.params.")
        let tokenizer = try Zonos3TextTokenizer(directory: directory, config: config.backbone)
        let voices = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: directory.appendingPathComponent("voices.json")))
        guard
            voices.values.allSatisfy({
                $0 == URL(fileURLWithPath: $0).lastPathComponent && $0.hasSuffix(".npz")
            })
        else {
            throw Zonos3Error.invalidCheckpoint("Invalid voice manifest paths")
        }
        eval(params.arrays)
        return Zonos3Model(
            config: config, directory: directory, params: params, tokenizer: tokenizer,
            voices: voices)
    }
    /// Extract a 192-dimensional embedding. Input is mono PCM at 44,100 Hz, capped at 60 seconds.
    public func extractSpeakerEmbedding(_ audio: MaterializedArray) throws -> MaterializedArray {
        try speakerEmbedding(audio).materialized()
    }
    private func speakerEmbedding(_ pcm: MLXArray) throws -> MLXArray {
        guard pcm.ndim == 1, pcm.size > 0, all(isFinite(pcm)).item(Bool.self) else {
            throw Zonos3Error.invalidInput("Reference must be finite mono PCM at 44,100 Hz")
        }
        if speakerEncoder == nil {
            speakerEncoder = try Zonos3SpeakerEncoder(
                directory.appendingPathComponent("speaker_encoder"))
        }
        return try speakerEncoder!.embed(pcm)
    }
    private func embedding(voice: String?, reference: MLXArray?, supplied: MLXArray?) throws
        -> MLXArray?
    {
        guard reference == nil || supplied == nil else {
            throw Zonos3Error.invalidInput(
                "Reference audio and speaker embedding are mutually exclusive")
        }
        let result: MLXArray
        if let reference {
            result = try speakerEmbedding(reference)
        } else if let supplied {
            result = supplied
        } else {
            let name = (voice ?? "jett").lowercased()
            if name == "none" { return nil }
            guard let filename = voiceFiles[name],
                let vector = try NumpyArchive.load(
                    url: directory.appendingPathComponent("voices/" + filename))["embedding"]
            else {
                throw Zonos3Error.invalidInput(
                    "Unknown voice \(name); available: \(voices.joined(separator: ", "))")
            }
            result = vector
        }
        guard
            result.shape == [config.backbone.speakerEmbeddingDim]
                || result.shape == [1, config.backbone.speakerEmbeddingDim],
            all(isFinite(result)).item(Bool.self)
        else {
            throw Zonos3Error.invalidInput(
                "Expected a finite \(config.backbone.speakerEmbeddingDim)-dimensional speaker embedding"
            )
        }
        return result.asType(.float32).reshaped(-1)
    }

    private func run(
        text: String, voice: String?, reference: MLXArray?, supplied: MLXArray?, language: String?,
        parameters p: GenerateParameters, options: Zonos3Options, interval: Double,
        emit: ((MLXArray) throws -> Void)? = nil
    ) throws -> (MLXArray, AudioGenerationInfo) {
        try Task.checkCancellation()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            interval.isFinite, interval > 0, p.temperature.isFinite, p.topP.isFinite,
            p.minP.isFinite,
            p.repetitionPenalty?.isFinite != false
        else { throw Zonos3Error.invalidInput("Empty text or invalid sampling settings") }
        let c = config.backbone
        let buckets = try Zonos3Generation.conditions(c, options: options)
        let speaker = try embedding(voice: voice, reference: reference, supplied: supplied)
        guard options.speakerCFGScale == 1 || speaker != nil else {
            throw Zonos3Error.invalidInput("Speaker guidance needs an embedding")
        }
        let text =
            options.textNormalization
            ? Zonos2Text.normalize(text, language: language ?? "en") : text
        let tokens = tokenizer.encode(text)
        let offset = speaker == nil ? 0 : 1
        let rows = MLXArray.full(
            [offset + c.conditionCount + tokens.count, c.nCodebooks + 1],
            values: MLXArray(c.audioPadId), dtype: .int32)
        rows[0..., c.nCodebooks] = MLXArray(c.textVocab)
        rows[(offset + c.conditionCount)..., c.nCodebooks] = MLXArray(tokens).asType(.int32)
        let silence = Zonos3Generation.delay(
            broadcast(
                MLXArray(c.silenceFrame).asType(.int32),
                to: [options.silencePrefillFrames, c.nCodebooks]), pad: c.audioPadId, flush: false)
        let prompt =
            silence.dim(0) > 0
            ? concatenated([
                rows,
                concatenated(
                    [
                        silence,
                        .full([silence.dim(0), 1], values: MLXArray(c.textVocab), dtype: .int32),
                    ], axis: -1),
            ]) : rows
        let budget = min(p.maxTokens ?? 2000, c.maxSeqlen - prompt.dim(0))
        guard budget > 0 else {
            throw Zonos3Error.invalidInput(
                "Token budget must be positive and leave context for audio")
        }
        let start = Date()
        let state = Zonos3LMState()
        var logits = try backbone(prompt, state: state, speaker: speaker, conditions: buckets)[-1]
        let twin = options.speakerCFGScale != 1 ? Zonos3LMState() : nil
        var twinLogits: MLXArray?
        if let twin {
            twinLogits = try backbone(
                prompt[1...], state: twin, conditions: c.conditionBucketCounts)[-1]
        }
        eval([logits] + state.arrays + (twin?.arrays ?? []) + [twinLogits].compactMap { $0 })
        let prefill = Date().timeIntervalSince(start)
        if codec == nil { codec = try Zonos3Codec(directory.appendingPathComponent("codec")) }
        let decoder = codec!
        let stream = emit == nil ? nil : Zonos3CodecStream(codec!)
        let history = MLXArray.zeros([silence.dim(0) + budget, c.nCodebooks], dtype: .int32)
        if silence.dim(0) > 0 { history[0 ..< silence.dim(0)] = silence }
        var stop: Int?
        var decoded = 0
        var generated = 0
        let chunkFrames = max(1, Int(min(interval * 15, Double(budget)).rounded()))
        for step in 0 ..< budget {
            try Task.checkCancellation()
            let t = silence.dim(0) + step
            let guided = twinLogits.map { $0 + options.speakerCFGScale * (logits - $0) } ?? logits
            let token = Zonos3Generation.sample(
                guided, config: c, step: step, history: history[0 ..< t],
                parameters: p, options: options, hasPrefix: silence.dim(0) > 0)
            history[t] = token
            let column = MLX.max(MLX.where(token .== c.eoaId, MLXArray(0 ..< c.nCodebooks), -1))
            eval([history, column] + state.arrays + (twin?.arrays ?? []))
            let lastStop = column.item(Int.self)
            if lastStop >= 0 { stop = min(stop ?? Int.max, max(0, t - lastStop)) }
            let ready = min(max(0, t + 2 - c.nCodebooks), stop ?? Int.max)
            if let stream, ready - decoded >= chunkFrames {
                let codes = Zonos3Generation.aligned(
                    history[0 ... t], codebooks: c.nCodebooks, stop: stop, start: decoded)
                let audio = try stream.decode(codes)
                decoded = ready
                if audio.size > 0 {
                    try emit?(
                        Zonos3Generation.output(audio, gain: decoder.gain(options.headroomDB)))
                }
            }
            generated = step + 1
            if let stop, t >= stop + c.nCodebooks - 1 { break }
            if step + 1 < budget {
                let next = concatenated([token, MLXArray([Int32(c.textVocab)])])[.newAxis]
                logits = try backbone(next, state: state)[-1]
                if let twin { twinLogits = try backbone(next, state: twin)[-1] }
            }
        }
        let codes = Zonos3Generation.aligned(
            history[0 ..< (silence.dim(0) + generated)], codebooks: c.nCodebooks, stop: stop)
        let audio: MLXArray
        if let stream {
            var tail: [MLXArray] = []
            if codes.dim(0) > decoded { tail.append(try stream.decode(codes[decoded...])) }
            tail.append(stream.flush())
            audio = Zonos3Generation.output(
                concatenated(tail), gain: decoder.gain(options.headroomDB))
            if audio.size > 0 { try emit?(audio) }
        } else {
            audio = Zonos3Generation.output(
                try decoder.decode(codes), gain: decoder.gain(options.headroomDB))
        }
        eval(audio)
        let elapsed = Date().timeIntervalSince(start) - prefill
        return (
            audio,
            AudioGenerationInfo(
                promptTokenCount: prompt.dim(0), generationTokenCount: generated,
                prefillTime: prefill, generateTime: elapsed,
                tokensPerSecond: Double(generated) / max(elapsed, 1e-9),
                peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
        )
    }

    public func generate(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        try await generate(
            text: text, voice: voice, refAudio: refAudio, language: language,
            generationParameters: generationParameters, options: .init())
    }
    public func generate(
        text: String, voice: String? = nil, refAudio: MaterializedArray? = nil,
        speakerEmbedding: MaterializedArray? = nil, language: String? = nil,
        generationParameters: GenerateParameters? = nil, options: Zonos3Options = .init()
    ) async throws -> MaterializedArray {
        try run(
            text: text, voice: voice, reference: refAudio, supplied: speakerEmbedding,
            language: language,
            parameters: generationParameters ?? defaultGenerationParameters, options: options,
            interval: 0.5
        ).0.materialized()
    }
    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, language: language,
            generationParameters: generationParameters, streamingInterval: 0.5, options: .init())
    }
    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
        language: String?,
        generationParameters: GenerateParameters, streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, language: language,
            generationParameters: generationParameters, streamingInterval: streamingInterval,
            options: .init())
    }
    public func generateStream(
        text: String, voice: String? = nil, refAudio: MaterializedArray? = nil,
        speakerEmbedding: MaterializedArray? = nil, language: String? = nil,
        generationParameters: GenerateParameters? = nil, streamingInterval: Double = 0.5,
        options: Zonos3Options = .init()
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    let (_, info) = try run(
                        text: text, voice: voice, reference: refAudio, supplied: speakerEmbedding,
                        language: language,
                        parameters: generationParameters ?? defaultGenerationParameters,
                        options: options, interval: streamingInterval
                    ) { audio in
                        if case .terminated = continuation.yield(.audio(audio)) {
                            throw CancellationError()
                        }
                    }
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
