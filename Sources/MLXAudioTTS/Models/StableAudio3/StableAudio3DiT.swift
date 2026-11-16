import Foundation
@preconcurrency import MLX
import MLXNN

enum StableAudio3Constants {
    static let latentChannels = 256
    static let conditioningDim = 768
    static let fourierDim = 256
    static let headDim = 64
}

struct StableAudio3DiTConfig {
    var embedDim = 1024
    var depth = 20
    var ffMult = 4.0
    var differential = false
}

private let localConditioningDim = 257
private let memoryTokenCount = 64

func stableAudio3FourierFeatures(_ t: MLXArray) -> MLXArray {
    let half = StableAudio3Constants.fourierDim / 2
    let ramp = MLXArray(0 ..< half).asType(.float32) / Float(half - 1)
    let logMin = Foundation.log(0.5)
    let logMax = Foundation.log(10000.0)
    let freqs = exp(ramp * Float(logMax - logMin) + Float(logMin))
    let args = t.expandedDimensions(axis: 1).asType(.float32) * freqs * Float(2) * Float(Double.pi)
    return concatenated([cos(args), sin(args)], axis: -1)
}

func stableAudio3Rope(_ x: MLXArray) -> MLXArray {
    MLXFast.RoPE(x, dimensions: 32, traditional: false, base: 10000, scale: 1, offset: 0)
}

let stableAudio3AttentionScale = Float(1.0 / sqrt(Double(StableAudio3Constants.headDim)))

func stableAudio3ToHeads(_ x: MLXArray) -> MLXArray {
    x.reshaped([x.dim(0), x.dim(1), -1, StableAudio3Constants.headDim]).transposed(0, 2, 1, 3)
}

func stableAudio3FromHeads(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(2), -1])
}

private func gatedLinear(_ layers: [UnaryLayer], _ x: MLXArray) -> MLXArray {
    layers[2](silu(layers[0](x)))
}

private func gatedMLP(_ first: Linear, _ last: Linear) -> [UnaryLayer] {
    [first, Identity(), last]
}

final class StableAudio3SelfAttention: Module {
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let partCount: Int

    init(dim: Int, differential: Bool) {
        partCount = differential ? 5 : 3
        _toQKV.wrappedValue = Linear(dim, partCount * dim, bias: false)
        _toOut.wrappedValue = Linear(dim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: StableAudio3Constants.headDim, eps: 1e-6)
        _kNorm.wrappedValue = RMSNorm(dimensions: StableAudio3Constants.headDim, eps: 1e-6)
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: stableAudio3Rope(qNorm(q)),
            keys: stableAudio3Rope(kNorm(k)),
            values: v,
            scale: stableAudio3AttentionScale,
            mask: .none
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(toQKV(x), parts: partCount, axis: -1).map(stableAudio3ToHeads)
        var out = attend(parts[0], parts[1], parts[2])
        if partCount == 5 {
            out = out - attend(parts[3], parts[4], parts[2])
        }
        return toOut(stableAudio3FromHeads(out))
    }
}

final class StableAudio3CrossAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let differential: Bool

    init(dim: Int, differential: Bool) {
        self.differential = differential
        _toQ.wrappedValue = Linear(dim, (differential ? 2 : 1) * dim, bias: false)
        _toKV.wrappedValue = Linear(dim, (differential ? 3 : 2) * dim, bias: false)
        _toOut.wrappedValue = Linear(dim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: StableAudio3Constants.headDim, eps: 1e-6)
        _kNorm.wrappedValue = RMSNorm(dimensions: StableAudio3Constants.headDim, eps: 1e-6)
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: qNorm(q),
            keys: kNorm(k),
            values: v,
            scale: stableAudio3AttentionScale,
            mask: .none
        )
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let out: MLXArray
        if differential {
            let queries = split(toQ(x), parts: 2, axis: -1).map(stableAudio3ToHeads)
            let keysValues = split(toKV(context), parts: 3, axis: -1).map(stableAudio3ToHeads)
            out = attend(queries[0], keysValues[0], keysValues[2]) - attend(queries[1], keysValues[1], keysValues[2])
        } else {
            let keysValues = split(toKV(context), parts: 2, axis: -1).map(stableAudio3ToHeads)
            out = attend(stableAudio3ToHeads(toQ(x)), keysValues[0], keysValues[1])
        }
        return toOut(stableAudio3FromHeads(out))
    }
}

final class StableAudio3GatedProjection: Module, UnaryLayer {
    @ModuleInfo var proj: Linear

    let sinusoidal: Bool

    init(dim: Int, inner: Int, sinusoidal: Bool = false) {
        self.sinusoidal = sinusoidal
        _proj.wrappedValue = Linear(dim, 2 * inner)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(proj(x), parts: 2, axis: -1)
        return parts[0] * (sinusoidal ? sin(parts[1] * Float(Double.pi)) : silu(parts[1]))
    }
}

final class StableAudio3FeedForward: Module {
    @ModuleInfo var ff: [UnaryLayer]

    init(dim: Int, inner: Int, sinusoidal: Bool = false) {
        _ff.wrappedValue = [StableAudio3GatedProjection(dim: dim, inner: inner, sinusoidal: sinusoidal), Identity(), Linear(inner, dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        ff[2](ff[0](x))
    }
}

final class StableAudio3DiTBlock: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: StableAudio3SelfAttention
    @ModuleInfo(key: "cross_attend_norm") var crossNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: StableAudio3CrossAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: RMSNorm
    @ModuleInfo var ff: StableAudio3FeedForward
    @ParameterInfo(key: "to_scale_shift_gate") var toScaleShiftGate: MLXArray
    @ModuleInfo(key: "to_local_embed") var toLocalEmbed: [UnaryLayer]

    init(config: StableAudio3DiTConfig) {
        let dim = config.embedDim
        _preNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        _selfAttention.wrappedValue = StableAudio3SelfAttention(dim: dim, differential: config.differential)
        _crossNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        _crossAttention.wrappedValue = StableAudio3CrossAttention(dim: dim, differential: config.differential)
        _ffNorm.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-5)
        _ff.wrappedValue = StableAudio3FeedForward(dim: dim, inner: Int(Double(dim) * config.ffMult))
        _toScaleShiftGate.wrappedValue = MLXArray.zeros([6 * dim])
        _toLocalEmbed.wrappedValue = gatedMLP(Linear(localConditioningDim, dim), Linear(dim, dim))
    }

    func callAsFunction(_ input: MLXArray, context: MLXArray, adaln: MLXArray, local: MLXArray) -> MLXArray {
        let gates = split((toScaleShiftGate + adaln).expandedDimensions(axis: 1), parts: 6, axis: -1)
        var h = preNorm(input) * (1 + gates[0]) + gates[1]
        var x = input + selfAttention(h) * sigmoid(1 - gates[2])
        x = x + crossAttention(crossNorm(x), context: context)
        let pad = MLXArray.zeros([x.dim(0), memoryTokenCount, x.dim(2)], dtype: x.dtype)
        x = x + concatenated([pad, gatedLinear(toLocalEmbed, local)], axis: 1)
        h = ffNorm(x) * (1 + gates[3]) + gates[4]
        return x + ff(h) * sigmoid(1 - gates[5])
    }
}

final class StableAudio3ContinuousTransformer: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "project_out") var projectOut: Linear
    @ParameterInfo(key: "memory_tokens") var memoryTokens: MLXArray
    @ModuleInfo(key: "global_cond_embedder") var globalCondEmbedder: [UnaryLayer]
    @ModuleInfo var layers: [StableAudio3DiTBlock]

    init(config: StableAudio3DiTConfig) {
        let dim = config.embedDim
        _projectIn.wrappedValue = Linear(StableAudio3Constants.latentChannels, dim, bias: false)
        _projectOut.wrappedValue = Linear(dim, StableAudio3Constants.latentChannels, bias: false)
        _memoryTokens.wrappedValue = MLXArray.zeros([memoryTokenCount, dim])
        _globalCondEmbedder.wrappedValue = gatedMLP(Linear(dim, dim), Linear(dim, 6 * dim))
        _layers.wrappedValue = (0 ..< config.depth).map { _ in StableAudio3DiTBlock(config: config) }
    }

    func callAsFunction(_ input: MLXArray, context: MLXArray, globalEmbed: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let memory = broadcast(memoryTokens.expandedDimensions(axis: 0), to: [batch] + memoryTokens.shape)
        var x = concatenated([memory, projectIn(input)], axis: 1)
        let adaln = gatedLinear(globalCondEmbedder, globalEmbed)
        let local = MLXArray.zeros([batch, length, localConditioningDim], dtype: x.dtype)
        for layer in layers {
            x = layer(x, context: context, adaln: adaln, local: local)
        }
        return projectOut(x[0..., memoryTokenCount..., 0...])
    }
}

final class StableAudio3DiT: Module {
    @ModuleInfo(key: "preprocess_conv") var preprocessConv: Conv1d
    @ModuleInfo(key: "postprocess_conv") var postprocessConv: Conv1d
    @ModuleInfo(key: "to_cond_embed") var toCondEmbed: [UnaryLayer]
    @ModuleInfo(key: "to_global_embed") var toGlobalEmbed: [UnaryLayer]
    @ModuleInfo(key: "to_timestep_embed") var toTimestepEmbed: [UnaryLayer]
    @ModuleInfo var transformer: StableAudio3ContinuousTransformer

    init(config: StableAudio3DiTConfig = .init()) {
        let channels = StableAudio3Constants.latentChannels
        let condDim = StableAudio3Constants.conditioningDim
        let dim = config.embedDim
        _preprocessConv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1, bias: false)
        _postprocessConv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1, bias: false)
        _toCondEmbed.wrappedValue = gatedMLP(Linear(condDim, dim, bias: false), Linear(dim, dim, bias: false))
        _toGlobalEmbed.wrappedValue = gatedMLP(Linear(condDim, dim, bias: false), Linear(dim, dim, bias: false))
        _toTimestepEmbed.wrappedValue = gatedMLP(
            Linear(StableAudio3Constants.fourierDim, dim), Linear(dim, dim))
        _transformer.wrappedValue = StableAudio3ContinuousTransformer(config: config)
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray, crossAttention: MLXArray, globalCond: MLXArray) -> MLXArray {
        let context = gatedLinear(toCondEmbed, crossAttention)
        let timestep = gatedLinear(toTimestepEmbed, stableAudio3FourierFeatures(t))
        let globalEmbed = gatedLinear(toGlobalEmbed, globalCond) + timestep
        let hidden = transformer(preprocessConv(x) + x, context: context, globalEmbed: globalEmbed)
        return postprocessConv(hidden) + hidden
    }
}
