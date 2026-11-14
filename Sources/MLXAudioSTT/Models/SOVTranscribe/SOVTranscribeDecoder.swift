// Gemma 4 text decoder of SOV-Transcribe-1, adapted from mlx-lm's gemma4_text (MIT).

import Foundation
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

/// RMS normalization without a learned scale, applied to Gemma 4 values.
final class SOVTranscribeRMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(eps: Float) { self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// Rotary embedding over the first `rotatedDimensions` of each head. Frequencies
/// keep the full head size as their denominator, as Gemma 4's "proportional" RoPE does.
final class SOVTranscribeRoPE: Module {
    let dimensions: Int
    private let _freqs: MLXArray?
    private let base: Float

    init(dimensions: Int, rope: SOVTranscribeTextConfig.RoPE) {
        self.dimensions = dimensions
        base = rope.ropeTheta ?? 10_000
        if rope.ropeType == "proportional" {
            let rotated = Int(Float(dimensions) * (rope.partialRotaryFactor ?? 1))
            let exponents = MLXArray(stride(from: 0, to: rotated, by: 2).map { Float($0) })
                / Float(dimensions)
            _freqs = concatenated([
                MLX.pow(MLXArray(base), exponents),
                MLXArray.full([(dimensions - rotated) / 2], values: MLXArray(Float.infinity)),
            ])
        } else {
            _freqs = nil
        }
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        if let freqs = _freqs {
            return MLXFast.RoPE(
                x, dimensions: dimensions, traditional: false, base: nil, scale: 1,
                offset: offset, freqs: freqs)
        }
        return MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: base, scale: 1, offset: offset)
    }
}

final class SOVTranscribeAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: SOVTranscribeRMSNormNoScale
    let rope: SOVTranscribeRoPE

    init(_ config: SOVTranscribeTextConfig, layerType: String) {
        let sliding = layerType == "sliding_attention"
        headDim = sliding ? config.headDim : config.globalHeadDim
        numHeads = config.numAttentionHeads
        // Full-attention layers reuse the keys as values ("K = V").
        let keysAreValues = config.attentionKEqV && !sliding
        numKVHeads =
            keysAreValues
            ? (config.numGlobalKeyValueHeads ?? config.numKeyValueHeads) : config.numKeyValueHeads
        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = keysAreValues ? nil : Linear(dim, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = SOVTranscribeRMSNormNoScale(eps: config.rmsNormEps)
        rope = SOVTranscribeRoPE(dimensions: headDim, rope: config.rope(for: layerType))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        var queries = qNorm(qProj(x).reshaped(b, l, numHeads, headDim)).transposed(0, 2, 1, 3)
        let rawKeys = kProj(x).reshaped(b, l, numKVHeads, headDim)
        let rawValues = vProj?(x).reshaped(b, l, numKVHeads, headDim) ?? rawKeys
        var keys = kNorm(rawKeys).transposed(0, 2, 1, 3)
        let values = vNorm(rawValues).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: 1, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class SOVTranscribeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: SOVTranscribeTextConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// Selects each token's experts. SOV runs the router in float32.
final class SOVTranscribeRouter: Module {
    let topK: Int
    let eps: Float
    let rootSize: Float
    @ModuleInfo var proj: Linear
    var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(_ config: SOVTranscribeTextConfig) {
        topK = config.topKExperts
        eps = config.rmsNormEps
        rootSize = pow(Float(config.hiddenSize), -0.5)
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        scale = MLXArray.ones([config.hiddenSize])
        _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let normed = MLXFast.rmsNorm(x.asType(.float32), weight: scale * rootSize, eps: eps)
        let scores = proj(normed)
        let experts = scores.dim(-1)
        let indices = argPartition(scores, kth: experts - topK, axis: -1)[
            .ellipsis, (experts - topK)...]
        let weights = softmax(takeAlong(scores, indices, axis: -1), axis: -1)
        return (indices, weights * perExpertScale[indices])
    }
}

/// The experts' gated GELU projections, gathered per selected expert.
final class SOVTranscribeSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: ExpertLinear
    @ModuleInfo(key: "up_proj") var upProj: ExpertLinear
    @ModuleInfo(key: "down_proj") var downProj: ExpertLinear

    init(_ config: SOVTranscribeTextConfig) {
        let (hidden, inner, experts) =
            (config.hiddenSize, config.moeIntermediateSize, config.numExperts)
        _gateProj.wrappedValue = ExpertLinear(input: hidden, output: inner, experts: experts)
        _upProj.wrappedValue = ExpertLinear(input: hidden, output: inner, experts: experts)
        _downProj.wrappedValue = ExpertLinear(input: inner, output: hidden, experts: experts)
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = x.expandedDimensions(axes: [-2, -3])
        var selected = indices
        let sorted = indices.size >= 64
        var inverse: MLXArray?
        if sorted {
            // Group tokens by expert so each expert's weights are read once.
            let flat = indices.reshaped(-1)
            let order = argSort(flat)
            inverse = argSort(order)
            input = input.reshaped(-1, 1, x.dim(-1))[order.floorDivide(indices.dim(-1))]
            selected = flat[order]
        }
        let gate = gateProj(input, indices: selected, sorted: sorted)
        let up = upProj(input, indices: selected, sorted: sorted)
        var y = downProj(geluApproximate(gate) * up, indices: selected, sorted: sorted)
        if let inverse { y = y[inverse].reshaped(indices.shape + [1, x.dim(-1)]) }
        return y.squeezed(axis: -2)
    }
}

final class SOVTranscribeExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SOVTranscribeSwitchGLU

    init(_ config: SOVTranscribeTextConfig) {
        _switchGLU.wrappedValue = SOVTranscribeSwitchGLU(config)
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let y = switchGLU(x, indices: indices)
        // Each weighted contribution returns to the hidden dtype before the sum,
        // as in the reference implementation.
        return (y * weights.expandedDimensions(axis: -1)).asType(x.dtype).sum(axis: -2)
    }
}

final class SOVTranscribeDecoderLayer: Module {
    let isSliding: Bool
    @ModuleInfo(key: "self_attn") var selfAttention: SOVTranscribeAttention
    @ModuleInfo var mlp: SOVTranscribeMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardLayerNorm: RMSNorm
    @ModuleInfo var router: SOVTranscribeRouter?
    @ModuleInfo var experts: SOVTranscribeExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedForwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedForwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedForwardLayerNorm2: RMSNorm?
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: SOVTranscribeTextConfig, layerType: String) {
        isSliding = layerType == "sliding_attention"
        let (dim, eps) = (config.hiddenSize, config.rmsNormEps)
        _selfAttention.wrappedValue = SOVTranscribeAttention(config, layerType: layerType)
        _mlp.wrappedValue = SOVTranscribeMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _preFeedForwardLayerNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postFeedForwardLayerNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        if config.enableMoeBlock {
            _router.wrappedValue = SOVTranscribeRouter(config)
            _experts.wrappedValue = SOVTranscribeExperts(config)
            _postFeedForwardLayerNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
            _postFeedForwardLayerNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
            _preFeedForwardLayerNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        }
        _layerScalar.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = x + postAttentionLayerNorm(selfAttention(inputLayerNorm(x), mask: mask, cache: cache))
        let residual = h
        if let router, let experts, let postFeedForwardLayerNorm1, let postFeedForwardLayerNorm2,
            let preFeedForwardLayerNorm2
        {
            let dense = postFeedForwardLayerNorm1(mlp(preFeedForwardLayerNorm(h)))
            let (indices, weights) = router(h)
            let routed = postFeedForwardLayerNorm2(
                experts(preFeedForwardLayerNorm2(h), indices: indices, weights: weights))
            h = dense + routed
        } else {
            h = mlp(preFeedForwardLayerNorm(h))
        }
        h = residual + postFeedForwardLayerNorm(h)
        return h * layerScalar
    }
}

/// Gemma 4 decoder stack. SOV scales text embeddings before inserting audio, so
/// this model takes input embeddings and applies no embedding scale itself.
public final class SOVTranscribeLanguageModel: Module {
    let config: SOVTranscribeTextConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SOVTranscribeDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: SOVTranscribeTextConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = config.layerTypes.map { SOVTranscribeDecoderLayer(config, layerType: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Scaled text embeddings, rounded to the embedding dtype like the reference.
    func embed(_ ids: MLXArray) -> MLXArray {
        let x = embedTokens(ids)
        return x * MLXArray(sqrt(Float(config.hiddenSize))).asType(x.dtype)
    }

    func makeCache() -> [KVCache] {
        layers.map {
            $0.isSliding
                ? RotatingKVCache(maxSize: config.slidingWindow, keep: 0) as KVCache
                : KVCacheSimple()
        }
    }

    func callAsFunction(inputEmbeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = inputEmbeddings
        let fullIndex = layers.firstIndex { !$0.isSliding }
        let slidingIndex = layers.firstIndex { $0.isSliding }
        let fullMask = fullIndex.map { createAttentionMask(h: h, cache: cache[$0]) } ?? .none
        let slidingMask =
            slidingIndex.map {
                createAttentionMask(h: h, cache: cache[$0], windowSize: config.slidingWindow)
            } ?? .none
        for (layer, layerCache) in zip(layers, cache) {
            h = layer(h, mask: layer.isSliding ? slidingMask : fullMask, cache: layerCache)
        }
        return norm(h)
    }

    /// Logits of the last `keep` positions, soft-capped as in Gemma 4.
    func logits(_ hidden: MLXArray, keep: Int = 1) -> MLXArray {
        let h = keep > 0 ? hidden[0..., (hidden.dim(1) - keep)..., 0...] : hidden
        var logits = embedTokens.asLinear(h)
        if let cap = config.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }
}
