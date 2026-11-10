import Foundation

/// Metadata shipped in `mlx-community/ZONOS3-Mini/config.json`.
public struct Zonos3Configuration: Codable, Sendable {
    public var modelType: String
    public var backbone: Zonos3BackboneConfiguration
    public var tensors: [String: Zonos3TensorSpecification]

    static func load(_ url: URL) throws -> Self {
        let decoder = JSONDecoder()
        let config = try decoder.decode(Self.self, from: Data(contentsOf: url))
        try config.backbone.validate()
        return config
    }
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backbone
        case tensors
    }

}

public struct Zonos3TensorSpecification: Codable, Sendable {
    public var shape: [Int]
    public var matrix: Bool
    public var dtype: String
    public var bits: Int?
}

public struct Zonos3BackboneConfiguration: Codable, Sendable {
    public var nLayers: Int
    public var maxSeqlen: Int
    public var dim: Int
    public var headDim: Int
    public var nHeads: Int
    public var nKvHeads: Int
    public var ffnHidden: Int
    public var nExperts: Int
    public var moeTopk: Int
    public var nCodebooks: Int
    public var codebookSize: Int
    public var audioVocab: Int
    public var eoaId: Int
    public var audioPadId: Int
    public var spanEndId: Int?
    public var textVocab: Int
    public var textExtraTokens: Int
    public var gdnHeads: Int
    public var gdnHeadDim: Int
    public var gdnKernel: Int
    public var speakerEmbeddingDim: Int
    public var conditionCount: Int
    public var conditionBucketCounts: [Int]
    public var conditionNames: [String]
    public var conditionUnits: [String: String]
    public var conditionEdges: [String: [Double]]
    public var ropeTheta: Float
    public var normEps: Float
    public var qkEps: Float
    public var lossSoftcap: Float
    public var gdnPeriod: Int
    public var silenceFrame: [Int]
    public var textTokenizerSpec: Zonos3TokenizerSpecification

    func validate() throws {
        guard nLayers > 0, dim > 0, maxSeqlen > 0, headDim > 0,
            nKvHeads > 0, nHeads > 0, nHeads % nKvHeads == 0,
            gdnHeads > 0, gdnHeadDim > 0, gdnKernel > 1, gdnPeriod > 0,
            nExperts > 0, moeTopk > 0, moeTopk <= nExperts,
            nCodebooks > 0, silenceFrame.count == nCodebooks,
            silenceFrame.allSatisfy({ (0 ..< codebookSize).contains($0) }),
            eoaId == codebookSize, audioPadId == codebookSize + 1,
            audioVocab == codebookSize + 2, speakerEmbeddingDim > 0,
            conditionNames.count == conditionCount, conditionBucketCounts.count == conditionCount,
            normEps > 0, qkEps > 0, lossSoftcap > 0, ropeTheta > 0
        else { throw Zonos3Error.invalidConfiguration("Inconsistent backbone dimensions") }
        for (name, count) in zip(conditionNames, conditionBucketCounts) {
            guard count > 0, let edges = conditionEdges[name], edges.count == count - 1,
                edges.allSatisfy(\.isFinite), edges == edges.sorted()
            else { throw Zonos3Error.invalidConfiguration("Invalid condition: \(name)") }
        }
    }
    enum CodingKeys: String, CodingKey {
        case nLayers = "n_layers"
        case maxSeqlen = "max_seqlen"
        case dim
        case headDim = "head_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case ffnHidden = "ffn_hidden"
        case nExperts = "n_experts"
        case moeTopk = "moe_topk"
        case nCodebooks = "n_codebooks"
        case codebookSize = "codebook_size"
        case audioVocab = "audio_vocab"
        case eoaId = "eoa_id"
        case audioPadId = "audio_pad_id"
        case spanEndId = "span_end_id"
        case textVocab = "text_vocab"
        case textExtraTokens = "text_extra_tokens"
        case gdnHeads = "gdn_heads"
        case gdnHeadDim = "gdn_head_dim"
        case gdnKernel = "gdn_kernel"
        case speakerEmbeddingDim = "speaker_embedding_dim"
        case conditionCount = "condition_count"
        case conditionBucketCounts = "condition_bucket_counts"
        case conditionNames = "condition_names"
        case conditionUnits = "condition_units"
        case conditionEdges = "condition_edges"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case qkEps = "qk_eps"
        case lossSoftcap = "loss_softcap"
        case gdnPeriod = "gdn_period"
        case silenceFrame = "silence_frame"
        case textTokenizerSpec = "text_tokenizer_spec"
    }

}

public struct Zonos3TokenizerSpecification: Codable, Sendable {
    public var kind: String
    public var path: String
    public var sha256: String
    public var vocabSize: Int
    public var offset: Int
    public var bos: Int
    public var eos: Int
    enum CodingKeys: String, CodingKey {
        case kind
        case path
        case sha256
        case vocabSize = "vocab_size"
        case offset
        case bos
        case eos
    }

}

public enum Zonos3Error: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case invalidCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message),
            .invalidCheckpoint(let message):
            "ZONOS3: \(message)"
        }
    }
}

/// Controls specific to ZONOS3. Reference waveforms are mono PCM at 44,100 Hz.
public struct Zonos3Options: Sendable {
    public var speed: Float = 1
    public var speakerCFGScale: Float = 1
    public var conditionBuckets: [String: Int]?
    public var conditionValues: [String: Double]?
    public var minFrames: Int = 0
    public var silencePrefillFrames: Int = 3
    public var repetitionCodebooks: Int = 6
    public var textNormalization: Bool = true
    public var headroomDB: Float?
    public init() {}
}
