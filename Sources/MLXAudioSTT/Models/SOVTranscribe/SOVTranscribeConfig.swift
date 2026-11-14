import Foundation
import MLXAudioCore

/// Whisper large-v3 encoder settings of SOV-Transcribe-1.
public struct SOVTranscribeAudioConfig: Codable, Sendable {
    public var dModel: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFFNDim: Int
    public var numMelBins: Int
    public var maxSourcePositions: Int

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        encoderAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
        encoderFFNDim = try c.decodeIfPresent(Int.self, forKey: .encoderFFNDim) ?? 5120
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
    }
}

/// Gemma 4 mixture-of-experts decoder settings of SOV-Transcribe-1.
public struct SOVTranscribeTextConfig: Codable, Sendable {
    public struct RoPE: Codable, Sendable {
        public var ropeTheta: Float?
        public var partialRotaryFactor: Float?
        public var ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case ropeType = "rope_type"
        }
    }

    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numGlobalKeyValueHeads: Int?
    public var headDim: Int
    public var globalHeadDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var slidingWindow: Int
    public var maxPositionEmbeddings: Int
    public var attentionKEqV: Bool
    public var finalLogitSoftcapping: Float?
    public var enableMoeBlock: Bool
    public var numExperts: Int
    public var topKExperts: Int
    public var moeIntermediateSize: Int
    public var numKVSharedLayers: Int
    public var hiddenSizePerLayerInput: Int
    public var layerTypes: [String]
    public var ropeParameters: [String: RoPE]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKEqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numGlobalKeyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? true
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30
        enableMoeBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? true
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 8
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 704
        numKVSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            let pattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
            layerTypes = (0 ..< numHiddenLayers).map {
                ($0 + 1) % pattern == 0 ? "full_attention" : "sliding_attention"
            }
        }
        ropeParameters =
            try c.decodeIfPresent([String: RoPE].self, forKey: .ropeParameters) ?? [
                "full_attention": RoPE(
                    ropeTheta: 1_000_000, partialRotaryFactor: 0.25, ropeType: "proportional"),
                "sliding_attention": RoPE(
                    ropeTheta: 10_000, partialRotaryFactor: 1, ropeType: "default"),
            ]
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encodeIfPresent(numGlobalKeyValueHeads, forKey: .numGlobalKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(globalHeadDim, forKey: .globalHeadDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(attentionKEqV, forKey: .attentionKEqV)
        try c.encodeIfPresent(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try c.encode(enableMoeBlock, forKey: .enableMoeBlock)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(topKExperts, forKey: .topKExperts)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(numKVSharedLayers, forKey: .numKVSharedLayers)
        try c.encode(hiddenSizePerLayerInput, forKey: .hiddenSizePerLayerInput)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(ropeParameters, forKey: .ropeParameters)
    }

    func rope(for layerType: String) -> RoPE {
        ropeParameters[layerType] ?? RoPE()
    }
}

public struct SOVTranscribeConfig: Codable, Sendable {
    public var modelType: String
    public var audioConfig: SOVTranscribeAudioConfig
    public var textConfig: SOVTranscribeTextConfig
    public var audioTokenId: Int
    public var downsampleFactor: Int
    public var projectorHiddenAct: String
    public var tieWordEmbeddings: Bool
    public var eosTokenIds: [Int]
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case audioTokenId = "audio_token_id"
        case downsampleFactor = "downsample_factor"
        case projectorHiddenAct = "projector_hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case eosTokenIds = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "sovereignai_transcribe"
        audioConfig = try c.decode(SOVTranscribeAudioConfig.self, forKey: .audioConfig)
        textConfig = try c.decode(SOVTranscribeTextConfig.self, forKey: .textConfig)
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 258881
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 4
        projectorHiddenAct =
            try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        if let ids = try? c.decodeIfPresent([Int].self, forKey: .eosTokenIds) {
            eosTokenIds = ids
        } else {
            eosTokenIds = [try c.decodeIfPresent(Int.self, forKey: .eosTokenIds) ?? 1]
        }
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(audioConfig, forKey: .audioConfig)
        try c.encode(textConfig, forKey: .textConfig)
        try c.encode(audioTokenId, forKey: .audioTokenId)
        try c.encode(downsampleFactor, forKey: .downsampleFactor)
        try c.encode(projectorHiddenAct, forKey: .projectorHiddenAct)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(eosTokenIds, forKey: .eosTokenIds)
    }

    /// Audio tokens one 30-second encoder window contributes to the prompt.
    var tokensPerWindow: Int { audioConfig.maxSourcePositions / downsampleFactor }

    func validate() throws {
        let a = audioConfig
        let t = textConfig
        func fail(_ reason: String) -> Error {
            STTError.modelNotInitialized("Unsupported SOV-Transcribe configuration: \(reason)")
        }
        guard t.modelType == "gemma4_text" else { throw fail("the decoder must be Gemma 4") }
        guard t.numKVSharedLayers == 0, t.hiddenSizePerLayerInput == 0 else {
            throw fail("shared KV layers and per-layer inputs are not used")
        }
        guard t.layerTypes.count == t.numHiddenLayers,
            Set(t.layerTypes).isSubset(of: ["sliding_attention", "full_attention"])
        else { throw fail("expected one supported attention type per decoder layer") }
        guard !t.enableMoeBlock || (t.numExperts > 0 && (1 ... t.numExperts).contains(t.topKExperts))
        else { throw fail("top_k_experts must be in [1, num_experts]") }
        guard downsampleFactor > 0, a.maxSourcePositions % downsampleFactor == 0 else {
            throw fail("audio encoder length must be divisible by downsample_factor")
        }
        guard projectorHiddenAct == "gelu" else { throw fail("only the GELU projector is supported") }
        guard a.dModel > 0, a.encoderAttentionHeads > 0, a.dModel % a.encoderAttentionHeads == 0,
            t.numAttentionHeads > 0, t.numKeyValueHeads > 0,
            t.numAttentionHeads % t.numKeyValueHeads == 0,
            (0 ..< t.vocabSize).contains(audioTokenId)
        else { throw fail("inconsistent layer sizes") }
    }
}
