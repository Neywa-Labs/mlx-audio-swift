import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

final class SOVTranscribeInner: Module {
    @ModuleInfo(key: "audio_tower") var audioTower: SOVTranscribeAudioEncoder
    @ModuleInfo(key: "embed_audio") var embedAudio: SOVTranscribeAudioProjector
    @ModuleInfo(key: "language_model") var languageModel: SOVTranscribeLanguageModel

    init(_ config: SOVTranscribeConfig) {
        _audioTower.wrappedValue = SOVTranscribeAudioEncoder(config.audioConfig)
        _embedAudio.wrappedValue = SOVTranscribeAudioProjector(config)
        _languageModel.wrappedValue = SOVTranscribeLanguageModel(config.textConfig)
    }
}

/// Salesforce SOV-Transcribe-1: a Whisper large-v3 encoder and a Gemma 4
/// mixture-of-experts decoder that transcribe recordings of up to 30 minutes.
public final class SOVTranscribeModel: Module, STTGenerationModel {
    /// The instruction the reference implementation uses when none is given.
    public static let defaultPrompt =
        "Transcribe the audio verbatim, outputting only spoken words in sequence."

    public let config: SOVTranscribeConfig
    @ModuleInfo(key: "model") var model: SOVTranscribeInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    public var tokenizer: Tokenizer?
    public private(set) var eosTokenIds: Set<Int>

    public var sampleRate: Int { SOVTranscribeAudio.sampleRate }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 32768, temperature: 0, topP: 1)
    }

    var languageModel: SOVTranscribeLanguageModel { model.languageModel }

    public init(_ config: SOVTranscribeConfig) {
        self.config = config
        _model.wrappedValue = SOVTranscribeInner(config)
        _lmHead.wrappedValue =
            config.tieWordEmbeddings
            ? nil : Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        eosTokenIds = Set(config.eosTokenIds).union([106, 50])
    }

    // MARK: Prompt

    /// The full instruction: the prompt, then the optional language hint and vocabulary.
    public static func instruction(
        prompt: String?, language: String?, hotwords: [String]? = nil
    ) -> String {
        var text = prompt.flatMap { $0.isEmpty ? nil : $0 } ?? defaultPrompt
        if let language, !language.isEmpty {
            text += "\nThe spoken language is \(language)."
        }
        let terms = (hotwords ?? []).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !terms.isEmpty {
            text += "\nContextual vocabulary: \(terms.joined(separator: ", "))"
        }
        return text
    }

    /// The checkpoint's chat template rendered for one user turn with text then audio.
    /// Audio placeholders are inserted as ids so long recordings skip tokenization.
    func promptIds(instruction: String, audioTokens: Int) -> [Int] {
        guard let tokenizer else { preconditionFailure("SOV-Transcribe tokenizer is not loaded") }
        let prefix = tokenizer.encode(
            text: "<bos><|turn>user\n\(instruction)<|audio>", addSpecialTokens: false)
        let suffix = tokenizer.encode(
            text: "<audio|><turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
            addSpecialTokens: false)
        return prefix + Array(repeating: config.audioTokenId, count: audioTokens) + suffix
    }

    // MARK: Inputs

    /// Projected audio features `[tokens, hidden]` for all 30-second windows.
    func audioFeatures(_ audio: MLXArray) -> MLXArray {
        let features = SOVTranscribeAudio.features(
            audio, numMelBins: config.audioConfig.numMelBins)
        let width = config.audioConfig.dModel * config.downsampleFactor
        var outputs: [MLXArray] = []
        for window in 0 ..< features.dim(0) {
            let encoded = model.audioTower(features[window ..< (window + 1)])
            let projected = model.embedAudio(encoded.reshaped(-1, width))
            eval(projected)
            outputs.append(projected)
        }
        return concatenated(outputs, axis: 0)
    }

    func prepareInputs(_ audio: MLXArray, instruction: String) throws -> (ids: [Int], embeddings: MLXArray)
    {
        let audioTokens = SOVTranscribeAudio.windowCount(samples: audio.dim(0)) * config.tokensPerWindow
        let ids = promptIds(instruction: instruction, audioTokens: audioTokens)
        guard ids.count < config.textConfig.maxPositionEmbeddings else {
            throw STTError.invalidInput("Audio and prompt exceed the model context window")
        }
        let features = audioFeatures(audio)
        let idArray = MLXArray(ids.map(Int32.init))
        let isAudio = idArray .== Int32(config.audioTokenId)
        let text = languageModel.embed(
            MLX.where(isAudio, MLXArray(Int32(0)), idArray).expandedDimensions(axis: 0))
        let positions = clip(cumsum(isAudio.asType(.int32)) - 1, min: 0, max: features.dim(0) - 1)
        let audioEmbeddings = features.asType(text.dtype)[positions].expandedDimensions(axis: 0)
        let embeddings = MLX.where(
            isAudio.expandedDimensions(axes: [0, 2]), audioEmbeddings, text)
        eval(embeddings)
        return (ids, embeddings)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        guard let lmHead else { return languageModel.logits(hidden) }
        var logits = lmHead(hidden[0..., (hidden.dim(1) - 1)..., 0...])
        if let cap = config.textConfig.finalLogitSoftcapping { logits = tanh(logits / cap) * cap }
        return logits
    }

    // MARK: Generation

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generate(audio: audio, generationParameters: generationParameters, prompt: nil)
    }

    /// Transcribes `audio` (mono, 16 kHz). `prompt` replaces the default instruction;
    /// `generationParameters.language` and `hotwords` are appended to it.
    /// The audio and prompt must leave room for at least one generated token.
    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, hotwords: [String]? = nil, prefillStepSize: Int = 512
    ) -> STTOutput {
        let parameters = generationParameters ?? defaultGenerationParameters
        do {
            return try generateImpl(
                audio: audio, parameters: parameters,
                instruction: Self.instruction(
                    prompt: prompt, language: parameters.language, hotwords: hotwords),
                prefillStepSize: prefillStepSize, emit: nil)
        } catch {
            preconditionFailure(error.localizedDescription)
        }
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        generateStream(audio: audio, generationParameters: generationParameters, prompt: nil)
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, hotwords: [String]? = nil, prefillStepSize: Int = 512
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let parameters = generationParameters ?? defaultGenerationParameters
        let instruction = Self.instruction(
            prompt: prompt, language: parameters.language, hotwords: hotwords)
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return makeSTTStream { continuation in
            let result = try model.value.generateImpl(
                audio: input.value, parameters: parameters, instruction: instruction,
                prefillStepSize: prefillStepSize
            ) { continuation.yield(.token($0)) }
            try Task.checkCancellation()
            continuation.yield(.result(result))
        }
    }

    private func generateImpl(
        audio: MLXArray, parameters p: STTGenerateParameters, instruction: String,
        prefillStepSize: Int, emit: ((String) -> Void)?
    ) throws -> STTOutput {
        guard let tokenizer else { preconditionFailure("SOV-Transcribe tokenizer is not loaded") }
        precondition(prefillStepSize > 0 && p.maxTokens >= 0 && p.temperature >= 0)
        precondition(
            audio.ndim == 1 && audio.size > 0, "Expected a nonempty mono waveform sampled at 16 kHz")
        precondition(
            audio.dim(0) <= SOVTranscribeAudio.maximumSamples,
            "SOV-Transcribe-1 supports recordings up to 30 minutes")
        let start = Date()
        let (ids, embeddings) = try prepareInputs(audio, instruction: instruction)
        let maxTokens = min(p.maxTokens, config.textConfig.maxPositionEmbeddings - ids.count)
        let cache = languageModel.makeCache()
        var logits = MLXArray(0)
        var offset = 0
        while offset < ids.count {
            if Task.isCancelled { break }
            let end = min(offset + prefillStepSize, ids.count)
            logits = self.logits(
                languageModel(inputEmbeddings: embeddings[0..., offset ..< end, 0...], cache: cache))
            eval([logits] + cache.flatMap { $0.innerState() })
            offset = end
        }
        let prefillTime = Date().timeIntervalSince(start)
        let generationStart = Date()
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler() : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK)
        var tokens: [Int] = []
        var emitted = ""
        for _ in 0 ..< maxTokens where !Task.isCancelled {
            let token = sampler.sample(logits: logits[0, -1, 0...].asType(.float32)).item(Int.self)
            if eosTokenIds.contains(token) { break }
            tokens.append(token)
            if emit != nil || p.verbose {
                let decoded = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                if let delta = transcriptDelta(from: emitted, to: decoded) {
                    emit?(delta)
                    if p.verbose { print(delta, terminator: "") }
                    emitted = decoded
                }
            }
            logits = self.logits(
                languageModel(
                    inputEmbeddings: languageModel.embed(MLXArray([Int32(token)]).reshaped(1, 1)),
                    cache: cache))
            eval(logits)
        }
        let decoded = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
        if let tail = transcriptDelta(from: emitted, to: decoded, complete: true) {
            emit?(tail)
            if p.verbose { print(tail) }
        }
        let text = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        let generationTime = Date().timeIntervalSince(generationStart)
        let duration = Double(audio.dim(0)) / Double(sampleRate)
        return STTOutput(
            text: text, segments: text.isEmpty ? [] : [["start": 0.0, "end": duration, "text": text]],
            language: p.language, promptTokens: ids.count, generationTokens: tokens.count,
            totalTokens: ids.count + tokens.count,
            promptTps: Double(ids.count) / max(prefillTime, 0.001),
            generationTps: Double(tokens.count) / max(generationTime, 0.001),
            totalTime: Date().timeIntervalSince(start),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    // MARK: Loading

    /// Maps the official checkpoint and community MLX exports onto this module tree.
    public static func sanitize(
        weights: [String: MLXArray], tieWordEmbeddings: Bool = true
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (originalKey, originalValue) in weights {
            var key = originalKey
            var value = originalValue
            if key.hasPrefix("audio_tower.") || key.hasPrefix("embed_audio.")
                || key.hasPrefix("language_model.")
            {
                key = "model." + key
            }
            if key == "lm_head.weight" && tieWordEmbeddings { continue }
            if ["self_attn.rotary_emb", "input_max", "input_min", "output_max", "output_min"]
                .contains(where: key.contains)
            {
                continue
            }
            if key.hasSuffix(".experts.gate_up_proj") {
                // Transformers stores the gate and up projections as one tensor.
                let base = String(key.dropLast(".gate_up_proj".count)) + ".switch_glu"
                let parts = split(value, parts: 2, axis: -2)
                result[base + ".gate_proj.weight"] = contiguous(parts[0])
                result[base + ".up_proj.weight"] = contiguous(parts[1])
                continue
            }
            if key.hasSuffix(".experts.down_proj") {
                result[String(key.dropLast(".down_proj".count)) + ".switch_glu.down_proj.weight"] =
                    value
                continue
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                key = key.replacingOccurrences(
                    of: ".experts.\(projection).", with: ".experts.switch_glu.\(projection).")
            }
            if ["model.audio_tower.conv1.weight", "model.audio_tower.conv2.weight"].contains(key),
                value.ndim == 3, value.dim(1) != 3
            {
                value = value.transposed(0, 2, 1)
            }
            if key.contains(".router.") { value = value.asType(.float32) }
            result[key] = value
        }
        return result
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> SOVTranscribeModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> SOVTranscribeModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: data)
        try config.validate()
        let model = SOVTranscribeModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No SOV-Transcribe safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights, tieWordEmbeddings: config.tieWordEmbeddings)
        if let quantization = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil
                    ? nil : quantization.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await loadSTTTokenizer(from: directory)
        model.eosTokenIds.formUnion(generationEosTokenIds(in: directory))
        guard model.tokenizer?.convertTokenToId("<|audio|>") == config.audioTokenId else {
            throw STTError.modelNotInitialized("Checkpoint audio token does not match audio_token_id")
        }
        model.train(false)
        eval(model)
        return model
    }

    /// End-of-turn ids from the processor and generation configurations.
    static func generationEosTokenIds(in directory: URL) -> [Int] {
        var ids: [Int] = []
        for (file, key) in [
            ("processor_config.json", "generation_eos_token_ids"),
            ("generation_config.json", "eos_token_id"),
        ] {
            guard
                let data = try? Data(contentsOf: directory.appendingPathComponent(file)),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            if let list = json[key] as? [Int] { ids += list } else if let id = json[key] as? Int {
                ids.append(id)
            }
        }
        return ids
    }
}
