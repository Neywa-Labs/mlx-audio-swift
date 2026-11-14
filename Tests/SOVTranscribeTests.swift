import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private let tinySOVConfigJSON = #"""
    {"model_type":"sovereignai_transcribe","audio_token_id":30,"eos_token_id":1,"downsample_factor":4,
     "tie_word_embeddings":false,
     "audio_config":{"d_model":8,"encoder_layers":1,"encoder_attention_heads":2,"encoder_ffn_dim":16,
       "num_mel_bins":8,"max_source_positions":1500},
     "text_config":{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,
       "num_attention_heads":2,"num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":4,
       "global_head_dim":8,"vocab_size":32,"sliding_window":4,"num_experts":4,"top_k_experts":2,
       "moe_intermediate_size":8,"layer_types":["sliding_attention","full_attention"]}}
    """#

private func tinySOVConfig() throws -> SOVTranscribeConfig {
    try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(tinySOVConfigJSON.utf8))
}

private func writeSOVTokenizer(_ directory: URL) throws {
    var vocab = Dictionary(uniqueKeysWithValues: (4 ..< 23).map { ("t\($0)", $0) })
    vocab["<pad>"] = 0
    vocab["<eos>"] = 1
    vocab["<bos>"] = 2
    vocab["a"] = 3
    vocab["b"] = 31
    let special: [(String, Int)] = [
        ("<pad>", 0), ("<eos>", 1), ("<bos>", 2), ("<|turn>", 23), ("<turn|>", 24),
        ("<|audio>", 25), ("<audio|>", 26), ("<|channel>", 27), ("<channel|>", 28),
        ("<unk>", 29), ("<|audio|>", 30),
    ]
    for (token, id) in special { vocab[token] = id }
    let data: [String: Any] = [
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>"],
        "added_tokens": special.map { ["id": $0.1, "content": $0.0, "special": true] },
    ]
    try JSONSerialization.data(withJSONObject: data).write(
        to: directory.appendingPathComponent("tokenizer.json"))
    let config: [String: Any] = [
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", "bos_token": "<bos>",
        "eos_token": "<eos>",
    ]
    try JSONSerialization.data(withJSONObject: config).write(
        to: directory.appendingPathComponent("tokenizer_config.json"))
}

/// A tiny checkpoint whose decoder always predicts token 3 ("a"): the decoder
/// weights are zero, so the hidden state is the scaled token embedding.
private func writeTinySOV(_ directory: URL) throws {
    let config = try tinySOVConfig()
    let model = SOVTranscribeModel(config)
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    weights["model.language_model.embed_tokens.weight"] = .ones([32, 8])
    weights["model.language_model.norm.weight"] = .ones([8])
    weights["lm_head.weight"] = concatenated(
        [MLXArray.zeros([3, 8]), .ones([1, 8]), .zeros([28, 8])], axis: 0)
    for i in 0 ..< 2 { weights["model.language_model.layers.\(i).layer_scalar"] = .ones([1]) }
    // Transformers layout: unprefixed keys, fused expert projections, (out, in, kernel) convolutions.
    var source: [String: MLXArray] = [:]
    for (key, value) in weights {
        var name = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
        var tensor = value
        if name.hasSuffix("audio_tower.conv1.weight") || name.hasSuffix("audio_tower.conv2.weight") {
            tensor = tensor.transposed(0, 2, 1)
        }
        if name.contains(".experts.switch_glu.") {
            if name.contains("gate_proj") {
                let up = weights[key.replacingOccurrences(of: "gate_proj", with: "up_proj")]!
                name = name.replacingOccurrences(of: "switch_glu.gate_proj.weight", with: "gate_up_proj")
                tensor = concatenated([tensor, up], axis: -2)
            } else if name.contains("up_proj") {
                continue
            } else {
                name = name.replacingOccurrences(of: "switch_glu.down_proj.weight", with: "down_proj")
            }
        }
        source[name] = tensor
    }
    eval(Array(source.values))
    try MLX.save(arrays: source, url: directory.appendingPathComponent("model.safetensors"))
    try Data(tinySOVConfigJSON.utf8).write(to: directory.appendingPathComponent("config.json"))
    try Data(#"{"eos_token_id":[1,24]}"#.utf8).write(
        to: directory.appendingPathComponent("generation_config.json"))
    try writeSOVTokenizer(directory)
}

@Suite(.serialized)
struct SOVTranscribeTests {
    private func directory() throws -> URL {
        let result = FileManager.default.temporaryDirectory.appendingPathComponent(
            "sov-transcribe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: result, withIntermediateDirectories: true)
        return result
    }

    @Test func publishedDefaultsAndLayerPattern() throws {
        let json = #"{"audio_config":{},"text_config":{"model_type":"gemma4_text"},"eos_token_id":[1,106]}"#
        let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(json.utf8))
        try config.validate()
        let text = config.textConfig
        #expect(text.hiddenSize == 2816 && text.numHiddenLayers == 30)
        #expect(text.numExperts == 128 && text.topKExperts == 8 && text.moeIntermediateSize == 704)
        #expect(text.layerTypes.count == 30)
        #expect(text.layerTypes[5] == "full_attention" && text.layerTypes[4] == "sliding_attention")
        #expect(text.rope(for: "full_attention").ropeType == "proportional")
        #expect(config.eosTokenIds == [1, 106])
        #expect(config.audioTokenId == 258881)
        #expect(config.tokensPerWindow == 375)
        let roundTrip = try JSONDecoder().decode(
            SOVTranscribeConfig.self, from: JSONEncoder().encode(config))
        #expect(roundTrip.textConfig.layerTypes == text.layerTypes)

        var shared = config
        shared.textConfig.numKVSharedLayers = 2
        #expect(throws: STTError.self) { try shared.validate() }
        var other = config
        other.textConfig.modelType = "gemma3_text"
        #expect(throws: STTError.self) { try other.validate() }
    }

    @Test func instructionAppendsLanguageAndVocabulary() {
        #expect(
            SOVTranscribeModel.instruction(prompt: nil, language: nil)
                == SOVTranscribeModel.defaultPrompt)
        #expect(
            SOVTranscribeModel.instruction(
                prompt: "Transcribe.", language: "German", hotwords: [" MLX ", "", "Gemma"])
                == "Transcribe.\nThe spoken language is German.\nContextual vocabulary: MLX, Gemma")
    }

    @Test func audioIsSplitIntoThirtySecondWindowsAfterFeatureExtraction() {
        #expect(SOVTranscribeAudio.windowCount(samples: 1) == 1)
        #expect(SOVTranscribeAudio.windowCount(samples: 480_000) == 1)
        #expect(SOVTranscribeAudio.windowCount(samples: 480_001) == 2)
        let audio = MLXRandom.normal([31 * 16000]) * 0.1
        #expect(SOVTranscribeAudio.features(audio, numMelBins: 8).shape == [2, 3000, 8])
    }

    @Test func proportionalRoPERotatesOnlyItsShareOfEachHead() {
        let rope = SOVTranscribeRoPE(
            dimensions: 8,
            rope: .init(ropeTheta: 1_000_000, partialRotaryFactor: 0.25, ropeType: "proportional"))
        let x = MLXRandom.normal([1, 1, 3, 8])
        let y = rope(x, offset: 5)
        // Half-split RoPE pairs dimension i with i + 4, so a quarter of eight
        // dimensions is the pair (0, 4); the other dimensions keep their values.
        let rotated = MLXArray([Int32(0), 4])
        let kept = MLXArray([Int32(1), 2, 3, 5, 6, 7])
        #expect(allClose(y.take(kept, axis: -1), x.take(kept, axis: -1)).item(Bool.self))
        #expect(!allClose(y.take(rotated, axis: -1), x.take(rotated, axis: -1)).item(Bool.self))
    }

    @Test func routerPicksTopExpertsAndScalesTheirWeights() throws {
        let router = SOVTranscribeRouter(try tinySOVConfig().textConfig)
        try router.update(
            parameters: ModuleParameters.unflattened([
                "proj.weight": concatenated(
                    [MLXArray.zeros([2, 8]), .ones([1, 8]), .ones([1, 8]) * 2], axis: 0),
                "per_expert_scale": MLXArray([Float(1), 1, 0.5, 2]),
            ]), verify: .none)
        let (indices, weights) = router(MLXArray.ones([1, 1, 8]).asType(.bfloat16))
        #expect(Set(indices.asArray(Int32.self)) == [2, 3])
        #expect(weights.dtype == .float32)
        let byExpert = Dictionary(
            uniqueKeysWithValues: zip(indices.asArray(Int32.self), weights.asArray(Float.self)))
        #expect(abs(byExpert[2]! / 0.5 + byExpert[3]! / 2 - 1) < 1e-5)
        #expect(byExpert[3]! > byExpert[2]!)
    }

    @Test func sanitizationMapsTransformersLayout() throws {
        let fused = MLXArray(0 ..< 24).reshaped(2, 6, 2).asType(.float32)
        let sanitized = SOVTranscribeModel.sanitize(weights: [
            "language_model.layers.0.experts.gate_up_proj": fused,
            "language_model.layers.0.experts.down_proj": MLXArray.zeros([2, 2, 3]),
            "model.language_model.layers.1.experts.gate_proj.weight": MLXArray.zeros([2, 3, 2]),
            "language_model.layers.0.router.scale": MLXArray.ones([2]).asType(.bfloat16),
            "audio_tower.conv1.weight": MLXArray.zeros([4, 8, 3]),
            "language_model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "lm_head.weight": MLXArray.zeros([2, 2]),
        ])
        let prefix = "model.language_model.layers.0."
        let gate = try #require(sanitized[prefix + "experts.switch_glu.gate_proj.weight"])
        let up = try #require(sanitized[prefix + "experts.switch_glu.up_proj.weight"])
        #expect(arrayEqual(gate, fused[0..., ..<3, 0...]).item(Bool.self))
        #expect(arrayEqual(up, fused[0..., 3..., 0...]).item(Bool.self))
        #expect(sanitized[prefix + "experts.switch_glu.down_proj.weight"] != nil)
        #expect(sanitized["model.language_model.layers.1.experts.switch_glu.gate_proj.weight"] != nil)
        #expect(sanitized[prefix + "router.scale"]?.dtype == .float32)
        #expect(sanitized["model.audio_tower.conv1.weight"]?.shape == [4, 3, 8])
        #expect(sanitized.keys.allSatisfy { !$0.contains("rotary_emb") && $0 != "lm_head.weight" })
    }

    @Test func tinyCheckpointLoadsAndStreamsTheSameTranscript() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinySOV(dir)
        let loaded = try await STT.loadModel(modelRepo: dir.path)
        let model = try #require(loaded as? SOVTranscribeModel)
        #expect(model.eosTokenIds.isSuperset(of: [1, 24, 50, 106]))

        let ids = model.promptIds(instruction: "a", audioTokens: 375)
        #expect(ids.prefix(2) == [2, 23])
        #expect(ids.filter { $0 == 30 }.count == 375)
        #expect(ids.suffix(1) == [28])

        let audio = MLXRandom.normal([31 * 16000]) * 0.1
        let parameters = STTGenerateParameters(maxTokens: 3, temperature: 0, topP: 1)
        let result = model.generate(audio: audio, generationParameters: parameters, prompt: "a")
        #expect(result.text == "aaa")
        #expect(result.generationTokens == 3)
        #expect(result.promptTokens == model.promptIds(instruction: "a", audioTokens: 750).count)

        var streamed = ""
        var finals = 0
        for try await event in model.generateStream(
            audio: audio, generationParameters: parameters, prompt: "a")
        {
            switch event {
            case .token(let delta): streamed += delta
            case .info: break
            case .result(let output):
                finals += 1
                #expect(output.text == result.text)
            }
        }
        #expect(finals == 1)
        #expect(streamed == result.text)
    }

    @Test func modelTypeIsInferredFromRepositoryName() {
        #expect(STT.inferModelType(from: "nativ-community/SOV-Transcribe-1-4bit") == "sovereignai_transcribe")
    }

    @Test(arguments: [0, -1], [8, 1500])
    func promptMustLeaveRoomForGeneration(contextRemaining: Int, sourcePositions: Int) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSOVTokenizer(dir)
        let tokenizer = try await loadSTTTokenizer(from: dir)
        var config = try tinySOVConfig()
        // The 8-position encoder cannot accept 3000 mel frames, so these cases
        // also verify that context rejection happens before audio encoding.
        config.audioConfig.maxSourcePositions = sourcePositions
        let probe = SOVTranscribeModel(config)
        probe.tokenizer = tokenizer
        let promptTokens = probe.promptIds(instruction: "a", audioTokens: config.tokensPerWindow).count
        config.textConfig.maxPositionEmbeddings = promptTokens + contextRemaining
        let model = SOVTranscribeModel(config)
        model.tokenizer = tokenizer
        let audio = MLXArray.zeros([16000])

        let inputError = #expect(throws: STTError.self) {
            try model.prepareInputs(audio, instruction: "a")
        }
        #expect(
            inputError?.localizedDescription
                == "Invalid input: Audio and prompt exceed the model context window")

        var events = 0
        let streamError = await #expect(throws: STTError.self) {
            for try await _ in model.generateStream(audio: audio, prompt: "a") { events += 1 }
        }
        #expect(
            streamError?.localizedDescription
                == "Invalid input: Audio and prompt exceed the model context window")
        #expect(events == 0)
    }

    @Test func generationUsesTheRemainingContext() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinySOV(dir)
        let probe = try await SOVTranscribeModel.fromModelDirectory(dir)
        var config = probe.config
        let promptTokens = probe.promptIds(instruction: "a", audioTokens: config.tokensPerWindow).count
        config.textConfig.maxPositionEmbeddings = promptTokens + 1
        try JSONEncoder().encode(config).write(to: dir.appendingPathComponent("config.json"))
        let model = try await SOVTranscribeModel.fromModelDirectory(dir)
        let result = model.generate(
            audio: .zeros([16000]), generationParameters: .init(maxTokens: 3), prompt: "a")
        #expect(result.text == "a")
        #expect(result.generationTokens == 1)
        #expect(result.promptTokens == promptTokens)
    }
}

@Suite(.serialized)
struct SOVTranscribeCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["SOV_TRANSCRIBE_MODEL_DIR"] != nil))
    func publishedCheckpointTranscribesWithLanguagePrompt() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["SOV_TRANSCRIBE_MODEL_DIR"])
        let common = try await STT.loadModel(modelRepo: path)
        let model = try #require(common as? SOVTranscribeModel)
        let url = try #require(
            Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media"))
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16000)
        let parameters = STTGenerateParameters(maxTokens: 256, temperature: 0, topP: 1)
        let result = model.generate(
            audio: audio, generationParameters: parameters,
            prompt: "Transcribe the English audio into text, ensuring all punctuation marks are included.")
        #expect(result.text.lowercased().contains("intention"))
        var streamed = ""
        for try await event in common.generateStream(audio: audio, generationParameters: parameters) {
            if case .token(let delta) = event { streamed += delta }
        }
        #expect(streamed.lowercased().contains("intention"))
    }
}
