import Foundation
@preconcurrency import MLX
import MLXNN

private let hiddenSize = 768
private let intermediateSize = 2048
private let numHeads = 12
private let headDim = 64
private let numLayers = 12
private let vocabSize = 256_000
private let rmsNormEps: Float = 1e-6
private let softcap: Float = 50
private let attentionScale = Float(1.0 / Foundation.sqrt(64.0))

final class StableAudio3TextRMSNorm: Module {
    let eps: Float
    @ParameterInfo var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

final class StableAudio3TextAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let rope = RoPE(dimensions: headDim, traditional: false, base: 10_000, scale: 1)

    override init() {
        _query.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        _key.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        _value.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        _output.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let queries = rope(query(x).reshaped([batch, length, numHeads, headDim]).transposed(0, 2, 1, 3))
        let keys = rope(key(x).reshaped([batch, length, numHeads, headDim]).transposed(0, 2, 1, 3))
        let values = value(x).reshaped([batch, length, numHeads, headDim]).transposed(0, 2, 1, 3)

        var scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * attentionScale
        scores = tanh(scores / softcap) * softcap
        scores = MLX.where(
            attentionMask[0..., .newAxis, .newAxis, 0...],
            scores,
            MLXArray(-Float.infinity).asType(scores.dtype)
        )
        let probabilities = softmax(scores.asType(.float32), axis: -1).asType(queries.dtype)
        let outputs = matmul(probabilities, values)
        return output(outputs.transposed(0, 2, 1, 3).reshaped([batch, length, -1]))
    }
}

final class StableAudio3TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    override init() {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(geluApproximate(gate(x)) * up(x))
    }
}

final class StableAudio3TextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: StableAudio3TextAttention
    @ModuleInfo(key: "pre_self_attn_layernorm") var preAttentionNorm: StableAudio3TextRMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postAttentionNorm: StableAudio3TextRMSNorm
    @ModuleInfo var mlp: StableAudio3TextMLP
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: StableAudio3TextRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: StableAudio3TextRMSNorm

    override init() {
        _attention.wrappedValue = StableAudio3TextAttention()
        _preAttentionNorm.wrappedValue = StableAudio3TextRMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        _postAttentionNorm.wrappedValue = StableAudio3TextRMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        _mlp.wrappedValue = StableAudio3TextMLP()
        _preFeedForwardNorm.wrappedValue = StableAudio3TextRMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        _postFeedForwardNorm.wrappedValue = StableAudio3TextRMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
    }

    func callAsFunction(_ input: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let attended = input + postAttentionNorm(attention(preAttentionNorm(input), attentionMask: attentionMask))
        return attended + postFeedForwardNorm(mlp(preFeedForwardNorm(attended)))
    }
}

final class StableAudio3TextEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [StableAudio3TextLayer]
    @ModuleInfo var norm: StableAudio3TextRMSNorm

    override init() {
        _embedTokens.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        _layers.wrappedValue = (0 ..< numLayers).map { _ in StableAudio3TextLayer() }
        _norm.wrappedValue = StableAudio3TextRMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
    }

    func callAsFunction(_ inputIDs: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var hidden = embedTokens(inputIDs)
        hidden = hidden * Float(Foundation.sqrt(Double(hiddenSize)))
        for layer in layers {
            hidden = layer(hidden, attentionMask: attentionMask)
        }
        return norm(hidden)
    }
}
