@preconcurrency import MLX
import MLXNN

struct StableAudio3DecoderConfig {
    var dim = 768
    var depth = 6
    var sinusoidalBlocks = 0
    var convMapping = true
    var slidingWindow = false
}

private let patchChannels = 512
private let patchSize = 256
private let stride = 16
private let subChunk = stride + 1
private let effectiveChunk = 34
private let shift = effectiveChunk / 2
private let chunk = 8
private let overlap = 2
private let slidingWindowChunk = 128
private let slidingWindowOverlap = 8
private let windowLength = 3 * subChunk
private let feedForwardMult = 3

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

func stableAudio3WindowMask(dtype: DType) -> MLXArray {
    let q = MLXArray(0 ..< subChunk).expandedDimensions(axis: 1)
    let kv = MLXArray(0 ..< windowLength).expandedDimensions(axis: 0)
    let valid = (kv .>= q) .&& (kv .<= q + 2 * subChunk)
    return MLX.where(valid, MLXArray(Float(0)).asType(dtype), MLXArray(Float(-1e9)).asType(dtype))
}

final class StableAudio3DifferentialAttention: Module {
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: StableAudio3DynamicTanh
    @ModuleInfo(key: "k_norm") var kNorm: StableAudio3DynamicTanh

    private let slidingWindow: Bool

    init(dim: Int, slidingWindow: Bool = false) {
        self.slidingWindow = slidingWindow
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

    func attendWindowed(_ q0: MLXArray, _ k0: MLXArray, _ v: MLXArray, _ qDiff0: MLXArray, _ kDiff0: MLXArray) -> MLXArray {
        let headDim = StableAudio3Constants.headDim
        let q = stableAudio3Rope(qNorm(q0))
        let k = stableAudio3Rope(kNorm(k0))
        let qDiff = stableAudio3Rope(qNorm(qDiff0))
        let kDiff = stableAudio3Rope(kNorm(kDiff0))
        let batch = q.dim(0)
        let heads = q.dim(1)
        let length = q.dim(2)
        let groups = length / subChunk
        let paddedLength = length + 2 * subChunk
        let shape = [batch, heads, groups, windowLength, headDim]
        let strides = [heads * paddedLength * headDim, paddedLength * headDim, subChunk * headDim, headDim, 1]
        let windows = [k, kDiff, v].map { part in
            asStrided(padded(part, widths: [[0, 0], [0, 0], [subChunk, subChunk], [0, 0]]), shape, strides: strides)
                .transposed(0, 2, 1, 3, 4)
                .reshaped([batch * groups, heads, windowLength, headDim])
        }
        let positions = MLXArray(0 ..< groups).expandedDimensions(axis: 1) * subChunk
            + MLXArray(0 ..< windowLength).expandedDimensions(axis: 0)
        let boundary = MLX.where(
            (positions .>= subChunk) .&& (positions .< length + subChunk),
            MLXArray(Float(0)).asType(q.dtype),
            MLXArray(Float(-1e9)).asType(q.dtype)
        ).expandedDimensions(axis: 1)
        let mask = broadcast(
            (stableAudio3WindowMask(dtype: q.dtype) + boundary).expandedDimensions(axis: 0),
            to: [batch, groups, subChunk, windowLength]
        ).reshaped([batch * groups, 1, subChunk, windowLength])
        let grouped = [q, qDiff].map { part in
            part.reshaped([batch, heads, groups, subChunk, headDim])
                .transposed(0, 2, 1, 3, 4)
                .reshaped([batch * groups, heads, subChunk, headDim])
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: concatenated(grouped, axis: 1),
            keys: concatenated([windows[0], windows[1]], axis: 1),
            values: concatenated([windows[2], windows[2]], axis: 1),
            scale: stableAudio3AttentionScale,
            mask: .array(mask)
        )
        let parts = split(out, parts: 2, axis: 1)
        return (parts[0] - parts[1])
            .reshaped([batch, groups, heads, subChunk, headDim])
            .transposed(0, 2, 1, 3, 4)
            .reshaped([batch, heads, length, headDim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(toQKV(x), parts: 5, axis: -1).map(stableAudio3ToHeads)
        let out: MLXArray
        if slidingWindow, x.dim(1) > subChunk {
            out = attendWindowed(parts[0], parts[1], parts[2], parts[3], parts[4])
        } else {
            out = attend(parts[0], parts[1], parts[2]) - attend(parts[3], parts[4], parts[2])
        }
        return toOut(stableAudio3FromHeads(out))
    }
}

final class StableAudio3DecoderBlock: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: StableAudio3DynamicTanh
    @ModuleInfo(key: "self_attn") var selfAttention: StableAudio3DifferentialAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: StableAudio3DynamicTanh
    @ModuleInfo var ff: StableAudio3FeedForward

    init(dim: Int, sinusoidal: Bool, slidingWindow: Bool) {
        _preNorm.wrappedValue = StableAudio3DynamicTanh(dims: dim)
        _selfAttention.wrappedValue = StableAudio3DifferentialAttention(dim: dim, slidingWindow: slidingWindow)
        _ffNorm.wrappedValue = StableAudio3DynamicTanh(dims: dim)
        _ff.wrappedValue = StableAudio3FeedForward(dim: dim, inner: feedForwardMult * dim, sinusoidal: sinusoidal)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = input + selfAttention(preNorm(input))
        return x + ff(ffNorm(x))
    }
}

class StableAudio3Decoder: Module {
    @ParameterInfo(key: "running_std") var runningStd: MLXArray
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ParameterInfo(key: "new_tokens") var newTokens: MLXArray
    @ModuleInfo var transformers: [StableAudio3DecoderBlock]
    @ModuleInfo var mapping: Conv1d

    let config: StableAudio3DecoderConfig

    init(config: StableAudio3DecoderConfig = .init()) {
        self.config = config
        let dim = config.dim
        _runningStd.wrappedValue = MLXArray.ones([1])
        _projectIn.wrappedValue = Linear(StableAudio3Constants.latentChannels, dim)
        _newTokens.wrappedValue = MLXArray.zeros([1, 1, dim])
        _transformers.wrappedValue = (0 ..< config.depth).map { index in
            StableAudio3DecoderBlock(
                dim: dim, sinusoidal: config.depth - index < config.sinusoidalBlocks,
                slidingWindow: config.slidingWindow)
        }
        let kernel = config.convMapping ? 3 : 1
        _mapping.wrappedValue = Conv1d(inputChannels: dim, outputChannels: patchChannels, kernelSize: kernel, padding: kernel / 2)
    }

    private func runChunks(_ input: MLXArray, blocks: ArraySlice<StableAudio3DecoderBlock>, chunks: Int) -> MLXArray {
        let batch = input.dim(0)
        var x = input.reshaped([batch * chunks, effectiveChunk, config.dim])
        for block in blocks {
            x = block(x)
        }
        return x.reshaped([batch, chunks * effectiveChunk, config.dim])
    }

    private func runShiftedChunks(_ input: MLXArray) -> MLXArray {
        let middle = transformers.count / 2
        let chunks = input.dim(1) / effectiveChunk
        var x = runChunks(input, blocks: transformers[..<middle], chunks: chunks)
        x = concatenated([x[0..., 0 ..< shift, 0...], x, x[0..., (x.dim(1) - shift)..., 0...]], axis: 1)
        x = runChunks(x, blocks: transformers[middle...], chunks: chunks + 1)
        return x[0..., shift ..< (x.dim(1) - shift), 0...]
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let length = latents.dim(1)
        let dim = config.dim
        var x = projectIn(latents * runningStd)
        let tokens = broadcast(newTokens.expandedDimensions(axis: 0), to: [batch, length, stride, dim])
        x = concatenated([x.expandedDimensions(axis: 2), tokens], axis: 2)
        x = x.reshaped([batch, length * subChunk, dim])

        if config.slidingWindow {
            for block in transformers {
                x = block(x)
            }
        } else {
            x = runShiftedChunks(x)
        }

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
        let windows = config.slidingWindow
            ? (chunk: slidingWindowChunk, overlap: slidingWindowOverlap) : (chunk: chunk, overlap: overlap)
        if total > windows.chunk + 2 * windows.overlap {
            patches = decodeWindows(latents, chunk: windows.chunk, overlap: windows.overlap)
        } else if config.slidingWindow || total % 2 == 0 {
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
