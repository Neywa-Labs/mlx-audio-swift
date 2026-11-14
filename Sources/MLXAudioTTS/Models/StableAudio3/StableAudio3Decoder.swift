@preconcurrency import MLX
import MLXNN

private let dim = 768
private let numHeads = 12
private let feedForwardInner = 2304
private let patchChannels = 512
private let patchSize = 256
private let stride = 16
private let subChunk = stride + 1
private let effectiveChunk = 34
private let shift = effectiveChunk / 2
private let chunk = 8
private let overlap = 2

final class StableAudio3DynamicTanh: Module, UnaryLayer {
    @ParameterInfo var alpha: MLXArray
    @ParameterInfo var gamma: MLXArray
    @ParameterInfo var beta: MLXArray

    init(dims: Int) {
        _alpha.wrappedValue = MLXArray.ones([1])
        _gamma.wrappedValue = MLXArray.ones([dims])
        _beta.wrappedValue = MLXArray.zeros([dims])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gamma * tanh(alpha * x) + beta
    }
}

final class StableAudio3DifferentialAttention: Module {
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: StableAudio3DynamicTanh
    @ModuleInfo(key: "k_norm") var kNorm: StableAudio3DynamicTanh

    override init() {
        _toQKV.wrappedValue = Linear(dim, 5 * dim, bias: false)
        _toOut.wrappedValue = Linear(dim, dim, bias: false)
        _qNorm.wrappedValue = StableAudio3DynamicTanh(dims: StableAudio3Constants.headDim)
        _kNorm.wrappedValue = StableAudio3DynamicTanh(dims: StableAudio3Constants.headDim)
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
        let batch = x.dim(0)
        let length = x.dim(1)
        let parts = split(toQKV(x), parts: 5, axis: -1).map {
            $0.reshaped([batch, length, numHeads, StableAudio3Constants.headDim]).transposed(0, 2, 1, 3)
        }
        let out = attend(parts[0], parts[1], parts[2]) - attend(parts[3], parts[4], parts[2])
        return toOut(out.transposed(0, 2, 1, 3).reshaped([batch, length, dim]))
    }
}

final class StableAudio3DecoderBlock: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: StableAudio3DynamicTanh
    @ModuleInfo(key: "self_attn") var selfAttention: StableAudio3DifferentialAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: StableAudio3DynamicTanh
    @ModuleInfo var ff: StableAudio3FeedForward

    override init() {
        _preNorm.wrappedValue = StableAudio3DynamicTanh(dims: dim)
        _selfAttention.wrappedValue = StableAudio3DifferentialAttention()
        _ffNorm.wrappedValue = StableAudio3DynamicTanh(dims: dim)
        _ff.wrappedValue = StableAudio3FeedForward(dim: dim, inner: feedForwardInner)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = input + selfAttention(preNorm(input))
        return x + ff(ffNorm(x))
    }
}

final class StableAudio3Decoder: Module {
    @ParameterInfo(key: "running_std") var runningStd: MLXArray
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ParameterInfo(key: "new_tokens") var newTokens: MLXArray
    @ModuleInfo var transformers: [StableAudio3DecoderBlock]
    @ModuleInfo var mapping: Conv1d

    init(depth: Int = 6) {
        _runningStd.wrappedValue = MLXArray.ones([1])
        _projectIn.wrappedValue = Linear(StableAudio3Constants.latentChannels, dim)
        _newTokens.wrappedValue = MLXArray.zeros([1, 1, dim])
        _transformers.wrappedValue = (0 ..< depth).map { _ in StableAudio3DecoderBlock() }
        _mapping.wrappedValue = Conv1d(inputChannels: dim, outputChannels: patchChannels, kernelSize: 3, padding: 1)
    }

    private func runChunks(_ input: MLXArray, blocks: ArraySlice<StableAudio3DecoderBlock>, chunks: Int) -> MLXArray {
        let batch = input.dim(0)
        var x = input.reshaped([batch * chunks, effectiveChunk, dim])
        for block in blocks {
            x = block(x)
        }
        return x.reshaped([batch, chunks * effectiveChunk, dim])
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let length = latents.dim(1)
        var x = projectIn(latents * runningStd)
        let tokens = broadcast(newTokens.expandedDimensions(axis: 0), to: [batch, length, stride, dim])
        x = concatenated([x.expandedDimensions(axis: 2), tokens], axis: 2)
        x = x.reshaped([batch, length * subChunk, dim])

        let middle = transformers.count / 2
        let chunks = x.dim(1) / effectiveChunk
        x = runChunks(x, blocks: transformers[..<middle], chunks: chunks)
        x = concatenated([x[0..., 0 ..< shift, 0...], x, x[0..., (x.dim(1) - shift)..., 0...]], axis: 1)
        x = runChunks(x, blocks: transformers[middle...], chunks: chunks + 1)
        x = x[0..., shift ..< (x.dim(1) - shift), 0...]

        x = x.reshaped([batch * length, subChunk, dim])[0..., 1..., 0...]
        return mapping(x.reshaped([batch, length * stride, dim]))
    }

    private func decodeWindows(_ latents: MLXArray, chunk: Int, overlap: Int) -> MLXArray {
        let total = latents.dim(1)
        let kernel = chunk + 2 * overlap
        let first = chunk + overlap
        var pieces = [callAsFunction(latents[0..., ..<kernel])[0..., ..<(first * stride)]]
        var position = first
        while position + chunk + overlap <= total {
            let window = latents[0..., (position - overlap) ..< (position + chunk + overlap)]
            let piece = callAsFunction(window)[0..., (overlap * stride) ..< ((overlap + chunk) * stride)]
            eval(piece)
            pieces.append(piece)
            position += chunk
        }
        if position < total {
            let last = callAsFunction(latents[0..., (total - kernel)...])
            pieces.append(last[0..., (-(total - position) * stride)...])
        }
        return concatenated(pieces, axis: 1)
    }

    func decode(_ latents: MLXArray) -> MLXArray {
        let total = latents.dim(1)
        let patches: MLXArray
        if total > chunk + 2 * overlap {
            patches = decodeWindows(latents, chunk: chunk, overlap: overlap)
        } else if total % 2 == 0 {
            patches = callAsFunction(latents)
        } else if total > 6 {
            patches = decodeWindows(latents, chunk: 2, overlap: 2)
        } else {
            let even = concatenated([latents, latents[0..., (-1)..., 0...]], axis: 1)
            patches = callAsFunction(even)[0..., ..<(total * stride)]
        }
        let batch = patches.dim(0)
        let length = patches.dim(1)
        let audio = patches.reshaped([batch, length, 2, patchSize]).transposed(0, 1, 3, 2)
        return audio.reshaped([batch, length * patchSize, 2])
    }
}
