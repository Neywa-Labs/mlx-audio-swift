// Copyright (c) 2026 Zyphra, Inc. Adapted under the MIT License.
import Foundation
import MLX
import MLXFFT
import MLXFast
import MLXNN

func zonos3LayerNorm(_ x: MLXArray, _ p: Zonos3TensorStore, _ prefix: String, _ eps: Float)
    -> MLXArray
{
    MLXFast.layerNorm(x, weight: p.t(prefix + ".weight"), bias: p.t(prefix + ".bias"), eps: eps)
}

/// Causal ZAC: semantic/residual RVQ, MOSS transformers, ConvNeXt, dual iSTFT, sinc resampling.
final class Zonos3Codec {
    struct Stage {
        let outDim: Int, heads: Int, layers: Int, context: Int, patch: Int
    }
    let p: Zonos3TensorStore
    let hop: Int, internalHop: Int, codebooks: Int, codebookSize: Int
    let semDepth: Int, headDepth: Int, headHop: Int, nFFT: [Int]
    let maxLogMagnitude: Float, headroomDB: Float, lnEps: Float, convEps: Float
    let resampleOrig: Int, resampleNew: Int, resampleWidth: Int
    let stages: [Stage]

    init(_ directory: URL) throws {
        let file = try Zonos3Component(directory, architecture: "mossdual_istft")
        p = file.params
        func value<T>(_ key: String, _: T.Type = T.self) throws -> T {
            guard let value = file.metadata["mossdual_istft." + key] as? T else {
                throw Zonos3Error.invalidCheckpoint("Missing codec metadata: \(key)")
            }
            return value
        }
        hop = try value("hop")
        internalHop = try value("internal_hop")
        codebooks = try value("n_codebooks")
        codebookSize = try value("codebook_size")
        semDepth = try value("sem_head.depth")
        headDepth = try value("head.depth")
        headHop = try value("head.hop")
        nFFT = try value("head.n_fft")
        maxLogMagnitude = Float(try value("head.max_log_magnitude", Double.self))
        headroomDB = Float(try value("headroom_db", Double.self))
        lnEps = Float(try value("ln_eps", Double.self))
        convEps = Float(try value("convnext_eps", Double.self))
        resampleOrig = try value("resample.orig")
        resampleNew = try value("resample.new")
        resampleWidth = try value("resample.width")
        let outs: [Int] = try value("stage.out_dim")
        let heads: [Int] = try value("stage.n_head")
        let layers: [Int] = try value("stage.n_layer")
        let contexts: [Int] = try value("stage.context")
        let patches: [Int] = try value("stage.patch")
        guard try value("sample_rate", Int.self) == 44100, hop == 2940, internalHop == 2944,
            codebooks == 10, codebookSize == 4096, nFFT == [736, 2208], headHop == 184,
            resampleOrig == 736, resampleNew == 735, resampleWidth == 17,
            !outs.isEmpty,
            [heads.count, layers.count, contexts.count, patches.count].allSatisfy({
                $0 == outs.count
            })
        else { throw Zonos3Error.invalidCheckpoint("Inconsistent ZAC decoder dimensions") }
        stages = outs.indices.map {
            Stage(
                outDim: outs[$0], heads: heads[$0], layers: layers[$0], context: contexts[$0],
                patch: patches[$0])
        }
        eval(p.arrays)
    }

    func gain(_ db: Float?) -> Float { pow(10, -(db ?? headroomDB) / 20) }

    static func patchUp(_ x: MLXArray, _ patch: Int) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1) / patch, patch).transposed(0, 2, 1).reshaped(
            -1, x.dim(1) / patch)
    }

    func convnext(_ x: MLXArray, prefix: String, history: MLXArray?) -> (MLXArray, MLXArray) {
        let w = p.t(prefix + ".dw.weight")
        let n = x.dim(0)
        let k = w.dim(1)
        let sequence = concatenated([history ?? .zeros([k - 1, x.dim(1)]), x])
        var h = sequence[0 ..< n] * w[0..., 0]
        for j in 1 ..< k { h = h + sequence[j ..< (j + n)] * w[0..., j] }
        h = h + p.t(prefix + ".dw.bias")
        h = zonos3LayerNorm(h, p, prefix + ".norm", convEps)
        h = gelu(p.lin(prefix + ".pw1.weight", h, bias: prefix + ".pw1.bias"))
        h = p.lin(prefix + ".pw2.weight", h, bias: prefix + ".pw2.bias")
        return (x + h, sequence[(sequence.dim(0) - k + 1)...])
    }

    func moss(
        _ input: MLXArray, prefix: String, stage: Stage, position: Int,
        cache: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let n = input.dim(0)
        let d = input.dim(1)
        let hd = d / stage.heads
        var h = zonos3LayerNorm(input, p, prefix + ".norm1", lnEps)
        let qkv = p.lin(prefix + ".qkv.weight", h).reshaped(n, 3, stage.heads, hd).transposed(
            1, 2, 0, 3)
        let q = MLXFast.RoPE(
            qkv[0], dimensions: hd, traditional: true, base: 10000, scale: 1, offset: position)
        var k = MLXFast.RoPE(
            qkv[1], dimensions: hd, traditional: true, base: 10000, scale: 1, offset: position)
        var v = qkv[2]
        if let cache {
            k = concatenated([cache.0, k], axis: 1)
            v = concatenated([cache.1, v], axis: 1)
        }
        let lc = k.dim(1) - n
        // Bound temporary attention matrices for long nonstreaming utterances as well.
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: n, by: 512) {
            let end = min(n, start + 512)
            let left = max(0, lc + start - stage.context + 1)
            let diff =
                MLXArray(position + start ..< position + end)[0..., .newAxis]
                - MLXArray(position - lc + left ..< position + end)[.newAxis, 0...]
            let mask = (diff .>= 0) .&& (diff .< stage.context)
            pieces.append(
                MLXFast.scaledDotProductAttention(
                    queries: q[.newAxis, 0..., start ..< end, 0...],
                    keys: k[.newAxis, 0..., left ..< (lc + end), 0...],
                    values: v[.newAxis, 0..., left ..< (lc + end), 0...],
                    scale: pow(Float(hd), -0.5), mask: .array(mask))[0])
        }
        let o = concatenated(pieces, axis: 1).transposed(1, 0, 2).reshaped(n, d)
        let x = input + p.lin(prefix + ".wo.weight", o) * p.t(prefix + ".attn_scale")
        h = zonos3LayerNorm(x, p, prefix + ".norm2", lnEps)
        h = p.lin(prefix + ".ffn2.weight", gelu(p.lin(prefix + ".ffn1.weight", h)))
        let keep = min(stage.context - 1, k.dim(1))
        let next: (MLXArray, MLXArray)? =
            keep > 0
            ? (k[0..., (k.dim(1) - keep)...], v[0..., (v.dim(1) - keep)...]) : nil
        return (x + h * p.t(prefix + ".ffn_scale"), next)
    }

    func decode(_ codes: MLXArray) throws -> MLXArray {
        let stream = Zonos3CodecStream(self)
        let audio = try stream.decode(codes)
        let tail = stream.flush()
        return concatenated([audio, tail])
    }
}

final class Zonos3CodecStream {
    let decoder: Zonos3Codec
    var frames = 0
    var semHistory: [MLXArray?], headHistory: [MLXArray?]
    var kv: [[(MLXArray, MLXArray)?]]
    let head: Zonos3ISTFTStream
    let resampler: Zonos3ResamplerStream

    init(_ decoder: Zonos3Codec) {
        self.decoder = decoder
        semHistory = Array(repeating: nil, count: decoder.semDepth)
        headHistory = Array(repeating: nil, count: decoder.headDepth)
        kv = decoder.stages.map { Array(repeating: nil, count: $0.layers) }
        head = Zonos3ISTFTStream(
            nFFT: decoder.nFFT, hop: decoder.headHop,
            windows: (0 ..< 2).map { decoder.p.t("head.window.\($0)") },
            mixer: decoder.p.t("head.mixer.weight"), maxLogMagnitude: decoder.maxLogMagnitude)
        resampler = Zonos3ResamplerStream(
            kernel: decoder.p.t("resample.kernel"),
            orig: decoder.resampleOrig, new: decoder.resampleNew, width: decoder.resampleWidth)
    }

    func decode(_ codes: MLXArray) throws -> MLXArray {
        let d = decoder
        guard codes.ndim == 2, codes.dim(1) == d.codebooks else {
            throw Zonos3Error.invalidInput("Codes must be [frames, \(d.codebooks)]")
        }
        if codes.dim(0) == 0 { return .zeros([0]) }
        guard !MLX.any((codes .< 0) .|| (codes .>= d.codebookSize)).item(Bool.self) else {
            throw Zonos3Error.invalidInput("Codes contain non-codec tokens")
        }
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: codes.dim(0), by: 32) {
            try Task.checkCancellation()
            chunks.append(chunk(codes[start ..< min(start + 32, codes.dim(0))].asType(.int32)))
        }
        return concatenated(chunks)
    }

    private func chunk(_ ids: MLXArray) -> MLXArray {
        let d = decoder
        let p = d.p
        var x = p.t("sem.table")[ids[0..., 0]] + p.t("sem.bias")
        for b in 0 ..< d.semDepth {
            (x, semHistory[b]) = d.convnext(x, prefix: "sem_head.\(b)", history: semHistory[b])
        }
        x = p.lin("sem_head.out.weight", x, bias: "sem_head.out.bias")
        for cb in 1 ..< d.codebooks { x = x + p.t("res.\(cb - 1).table")[ids[0..., cb]] }
        x = x + p.t("res.bias")
        eval(x)
        var positionsPerFrame = 1
        for (s, hp) in d.stages.enumerated() {
            let prefix = "stage.\(s)"
            x = p.lin(prefix + ".in.weight", x)
            for layer in 0 ..< hp.layers {
                (x, kv[s][layer]) = d.moss(
                    x, prefix: prefix + ".l\(layer)", stage: hp,
                    position: frames * positionsPerFrame, cache: kv[s][layer])
                eval([x] + (kv[s][layer].map { [$0.0, $0.1] } ?? []))
            }
            if hp.outDim > 0 { x = p.lin(prefix + ".out.weight", x) }
            if hp.patch > 1 {
                x = Zonos3Codec.patchUp(x, hp.patch)
                positionsPerFrame *= hp.patch
            }
        }
        for b in 0 ..< d.headDepth {
            (x, headHistory[b]) = d.convnext(x, prefix: "head.\(b)", history: headHistory[b])
            eval(x)
        }
        x = zonos3LayerNorm(x, p, "head.norm", d.convEps)
        let y0 = p.lin("head.out.0.weight", x, bias: "head.out.0.bias")
        let y1 = p.lin("head.out.1.weight", x, bias: "head.out.1.bias")
        eval([y0, y1] + (semHistory + headHistory).compactMap { $0 })
        let out = resampler.push(head.push([y0, y1]))
        eval([out, resampler.buffer] + head.ola + head.envelope + head.mixInput)
        frames += ids.dim(0)
        return out
    }

    func flush() -> MLXArray {
        let out = concatenated([resampler.push(head.flush()), resampler.flush()])
        eval(out)
        return out
    }
}

/// Finalize only samples no future iSTFT frame or FIR tap can change.
final class Zonos3ISTFTStream {
    let nFFT: [Int], hop: Int, windows: [MLXArray], mixer: MLXArray, maxLogMagnitude: Float
    let half: Int
    var ola = [MLXArray.zeros([0]), MLXArray.zeros([0])]
    var envelope = [MLXArray.zeros([0]), MLXArray.zeros([0])]
    var base = [0, 0]
    var mixInput: [MLXArray]
    var frames = 0
    var mixBase = 0
    var done = false

    init(nFFT: [Int], hop: Int, windows: [MLXArray], mixer: MLXArray, maxLogMagnitude: Float) {
        self.nFFT = nFFT
        self.hop = hop
        self.windows = windows
        self.mixer = mixer
        self.maxLogMagnitude = maxLogMagnitude
        half = (mixer.dim(1) - 1) / 2
        mixInput = [.zeros([half]), .zeros([half])]
    }

    func push(_ spectra: [MLXArray]) -> MLXArray {
        precondition(!done)
        let cols = spectra[0].dim(0)
        if cols == 0 { return .zeros([0]) }
        for i in 0 ..< 2 {
            let n = nFFT[i]
            let bins = n / 2 + 1
            let r = n / hop
            let spec = spectra[i]
            let mag = exp(minimum(spec[0..., 0 ..< bins], maxLogMagnitude))
            let phase = spec[0..., bins ..< (2 * bins)]
            let edge = ((MLXArray(0 ..< bins) .> 0) .&& (MLXArray(0 ..< bins) .< bins - 1)).asType(
                .float32)
            let complex =
                mag * cos(phase) + MLXArray(real: Float(0), imaginary: Float(1)) * mag * sin(phase)
                * edge
            let inverse = MLXFFT.irfft(complex, n: n) * windows[i]
            let parts = inverse.reshaped(cols, r, hop)
            let win2 = (windows[i] * windows[i]).reshaped(r, hop)
            let add = MLXArray.zeros([cols + r - 1, hop])
            let env = MLXArray.zeros([cols + r - 1, hop])
            for j in 0 ..< r {
                add[j ..< (j + cols)] = add[j ..< (j + cols)] + parts[0..., j]
                env[j ..< (j + cols)] = env[j ..< (j + cols)] + win2[j]
            }
            let offset = frames * hop - base[i]
            let need = offset + add.size
            if ola[i].size < need {
                let zeros = MLXArray.zeros([need - ola[i].size])
                ola[i] = concatenated([ola[i], zeros])
                envelope[i] = concatenated([envelope[i], zeros])
            }
            // Never mutate a wrapper shared with another buffer.
            let current = ola[i].reshaped(-1)
            let currentEnv = envelope[i].reshaped(-1)
            current[offset ..< need] = current[offset ..< need] + add.reshaped(-1)
            currentEnv[offset ..< need] = currentEnv[offset ..< need] + env.reshaped(-1)
            ola[i] = current
            envelope[i] = currentEnv
        }
        frames += cols
        for i in 0 ..< 2 { finalize(i, through: frames * hop) }
        return emit()
    }

    private func finalize(_ i: Int, through position: Int) {
        let n = position - base[i]
        guard n > 0 else { return }
        let skip = min(n, max(0, nFFT[i] / 2 - base[i]))
        mixInput[i] = concatenated([
            mixInput[i], ola[i][skip ..< n] / maximum(envelope[i][skip ..< n], 1e-12),
        ])
        ola[i] = ola[i][n...]
        envelope[i] = envelope[i][n...]
        base[i] = position
    }

    private func emit() -> MLXArray {
        let k = mixer.dim(1)
        let length = min(mixInput[0].size, mixInput[1].size) - k + 1
        guard length > 0 else { return .zeros([0]) }
        let take = length + k - 1
        let x = stacked(mixInput.map { $0[0 ..< take] }, axis: -1)[.newAxis]
        let out = conv1d(x, mixer.T[.newAxis])[0, 0..., 0]
        mixInput = mixInput.map { $0[length...] }
        mixBase += length
        return out
    }

    func flush() -> MLXArray {
        guard !done, frames > 0 else { return .zeros([0]) }
        done = true
        for i in 0 ..< 2 {
            finalize(i, through: base[i] + ola[i].size)
            let keep = half + frames * hop - mixBase
            var x = mixInput[i]
            if x.size >= keep {
                x = x[0 ..< keep]
            } else {
                x = concatenated([x, .zeros([keep - x.size])])
            }
            mixInput[i] = concatenated([x, .zeros([half])])
        }
        return emit()
    }
}

final class Zonos3ResamplerStream {
    let kernel: MLXArray, orig: Int, new: Int, width: Int
    var buffer = MLXArray.zeros([0])
    var base = 0, inputCount = 0, nextBlock = 0
    init(kernel: MLXArray, orig: Int, new: Int, width: Int) {
        self.kernel = kernel
        self.orig = orig
        self.new = new
        self.width = width
    }
    func push(_ x: MLXArray) -> MLXArray {
        buffer = concatenated([buffer, x])
        inputCount += x.size
        return run(flush: false)
    }
    func flush() -> MLXArray { run(flush: true) }
    private func run(flush: Bool) -> MLXArray {
        let taps = 2 * width + orig
        let total = (new * inputCount + orig - 1) / orig
        let reach = inputCount + width - taps
        let end = flush ? (total + new - 1) / new : (reach >= 0 ? reach / orig + 1 : 0)
        var out = MLXArray.zeros([0])
        if end > nextBlock {
            let lo = nextBlock * orig - width
            let hi = (end - 1) * orig - width + taps
            let segment = MLXArray.zeros([hi - lo])
            let a = max(lo, 0)
            let b = min(hi, inputCount)
            if b > a { segment[(a - lo) ..< (b - lo)] = buffer[(a - base) ..< (b - base)] }
            let windows = asStrided(segment, [end - nextBlock, taps], strides: [orig, 1])
            out = matmul(windows, kernel.T).reshaped(-1)
            if flush { out = out[0 ..< max(0, total - nextBlock * new)] }
            nextBlock = end
        }
        let keep = min(inputCount, max(0, nextBlock * orig - width))
        if keep > base {
            buffer = buffer[(keep - base)...]
            base = keep
        }
        return out
    }
}
