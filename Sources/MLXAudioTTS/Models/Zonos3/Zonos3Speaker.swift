// Copyright (c) 2026 Zyphra, Inc. Adapted under the MIT License.
import Accelerate
import Foundation
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

/// MECT-B2 speaker encoder with the checkpoint's Kaldi frontend.
final class Zonos3SpeakerEncoder {
    let p: Zonos3TensorStore
    let window: [Double], melBanks: [Double], resampleKernel: MLXArray
    let preemphasis: Double, waveformScale: Double, logFloor: Double
    let lnEps: Float, varianceFloor: Float
    var grouped: [String: MLXArray] = [:]

    init(_ directory: URL) throws {
        let file = try Zonos3Component(directory, architecture: "mect_spk")
        p = file.params
        let m = file.metadata
        guard m["mect_spk.backbone"] as? String == "MECT_B2",
            m["mect_spk.embedding_length"] as? Int == 192,
            m["mect_spk.n_fft"] as? Int == 512, m["mect_spk.n_mels"] as? Int == 80,
            m["mect_spk.frame_length"] as? Int == 400, m["mect_spk.frame_shift"] as? Int == 160
        else { throw Zonos3Error.invalidCheckpoint("Expected MECT-B2 speaker encoder") }
        func number(_ key: String) throws -> Double {
            guard let n = m["mect_spk." + key] as? Double else {
                throw Zonos3Error.invalidCheckpoint("Missing speaker metadata: \(key)")
            }
            return n
        }
        preemphasis = try number("preemphasis")
        waveformScale = try number("waveform_scale")
        logFloor = try number("log_floor")
        lnEps = Float(try number("ln_eps"))
        varianceFloor = Float(try number("pool_var_floor"))
        let host = try NumpyArchive.load(url: directory.appendingPathComponent("frontend.npz"))
        guard let win = host["frontend.window"], let mel = host["frontend.mel_banks"],
            let kernel = host["frontend.resample_44100"], win.shape == [400],
            mel.shape == [80, 257], kernel.shape == [160, 475]
        else {
            throw Zonos3Error.invalidCheckpoint("Missing speaker frontend arrays")
        }
        window = win.asType(.float64, stream: .cpu).asArray(Double.self)
        melBanks = mel.asType(.float64, stream: .cpu).asArray(Double.self)
        resampleKernel = kernel.asType(.float64, stream: .cpu)
        for name in p.tables.keys.filter({ $0.hasSuffix(".w0") }) {
            let prefix = String(name.dropLast(3))
            var slots: [MLXArray] = []
            while let weight = p.tables[prefix + ".w\(slots.count)"] { slots.append(weight) }
            grouped[prefix] = concatenated(slots, axis: 1).transposed(0, 2, 3, 1)
        }
        eval(p.arrays + Array(grouped.values))
    }

    static func resample(_ pcm: MLXArray, kernel: MLXArray, orig: Int, new: Int, width: Int)
        -> MLXArray
    {
        let length = pcm.size
        let total = (new * length + orig - 1) / orig
        if total == 0 { return .zeros([0]) }
        let taps = 2 * width + orig
        let blocks = (total + new - 1) / new
        let right = max(0, (blocks - 1) * orig + taps - width - length)
        let x = pcm.asType(.float64, stream: .cpu)
        let padded = concatenated(
            [
                .zeros([width], dtype: .float64, stream: .cpu), x,
                .zeros([right], dtype: .float64, stream: .cpu),
            ], stream: .cpu)
        let windows = asStrided(padded, [blocks, taps], strides: [orig, 1], stream: .cpu)
        return matmul(windows, kernel.transposed(stream: .cpu), stream: .cpu)
            .reshaped(-1, stream: .cpu)[0 ..< total, stream: .cpu].asType(.float32, stream: .cpu)
    }

    /// Double-precision FFT and accumulation preserve the published frontend's numerical behavior.
    func fbank(_ waveform: [Float]) throws -> MLXArray {
        let count = min(waveform.count, 960000)
        guard count >= 400 else {
            throw Zonos3Error.invalidInput("Speaker audio is shorter than 25 ms")
        }
        guard let fft = vDSP_DFT_zop_CreateSetupD(nil, 512, .FORWARD) else {
            throw Zonos3Error.invalidInput("Could not initialize speaker FFT")
        }
        defer { vDSP_DFT_DestroySetupD(fft) }
        let frames = 1 + (count - 400) / 160
        var output = [Float](repeating: 0, count: frames * 80)
        var real = [Double](repeating: 0, count: 512)
        var imag = real
        var fftReal = real
        var fftImag = real
        for t in 0 ..< frames {
            let start = t * 160
            let mean =
                waveform[start ..< (start + 400)].reduce(0.0) { $0 + Double($1) * waveformScale }
                / 400
            var previous = Double(waveform[start]) * waveformScale - mean
            for j in 0 ..< 400 {
                let value = Double(waveform[start + j]) * waveformScale - mean
                real[j] = (value - preemphasis * previous) * window[j]
                previous = value
            }
            vDSP_DFT_ExecuteD(fft, real, imag, &fftReal, &fftImag)
            var power = [Double](repeating: 0, count: 257)
            for j in 0 ..< 257 { power[j] = fftReal[j] * fftReal[j] + fftImag[j] * fftImag[j] }
            for mel in 0 ..< 80 {
                var energy = 0.0
                for j in 0 ..< 257 { energy += power[j] * melBanks[mel * 257 + j] }
                output[t * 80 + mel] = Float(log(max(energy, logFloor)))
            }
        }
        // Reference rounds fbank to FP32 before double-precision mean normalization.
        var mean = [Double](repeating: 0, count: 80)
        for t in 0 ..< frames {
            for m in 0 ..< 80 { mean[m] += Double(output[t * 80 + m]) / Double(frames) }
        }
        for t in 0 ..< frames {
            for m in 0 ..< 80 { output[t * 80 + m] = Float(Double(output[t * 80 + m]) - mean[m]) }
        }
        return MLXArray(output, [frames, 80])
    }

    private func bn(_ x: MLXArray, _ prefix: String) -> MLXArray {
        x * p.t(prefix + ".scale") + p.t(prefix + ".shift")
    }
    private func linear(_ x: MLXArray, _ prefix: String, bias: Bool = true) -> MLXArray {
        let w = p.t(prefix + ".weight")
        let y = matmul(x, w.reshaped(w.dim(0), -1).T)
        return bias ? y + p.t(prefix + ".bias") : y
    }
    private func ln(_ x: MLXArray, _ prefix: String) -> MLXArray {
        zonos3LayerNorm(x, p, prefix, lnEps)
    }
    private func groupConv(_ x: MLXArray, _ prefix: String, _ sf: Int, _ st: Int) -> MLXArray {
        let w = grouped[prefix]!
        return conv2d(x, w, stride: .init((sf, st)), padding: 1, groups: x.dim(-1) / w.dim(-1))
    }
    private func residual(_ x: MLXArray, _ q: String, _ sf: Int, _ st: Int) -> MLXArray {
        var t = groupConv(x, q + "conv1", sf, st)
        t = maximum(bn(linear(t, q + "pw1"), q + "bn1"), 0)
        t = groupConv(t, q + "conv2", 1, 1)
        t = maximum(bn(linear(t, q + "pw2"), q + "bn2"), 0)
        var sc = x
        if p.tables[q + "sc.weight"] != nil {
            sc = bn(
                linear(x[0..., .stride(by: sf), .stride(by: st), 0...], q + "sc", bias: false),
                q + "sc.bn")
        }
        return t + sc
    }
    private func attention(_ h: MLXArray, _ prefix: String) -> MLXArray {
        let n = h.dim(0)
        let heads = 4
        let d = 16
        func project(_ name: String) -> MLXArray {
            linear(h, prefix + "attn." + name).reshaped(n, heads, d).transposed(1, 0, 2)[.newAxis]
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: project("q") / 4,
            keys: project("k"), values: project("v"), scale: 1, mask: .none)[0]
        return linear(out.transposed(1, 0, 2).reshaped(n, 64), prefix + "attn.o")
    }
    private func newGELU(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1 + tanh(sqrt(Float(2 / Double.pi)) * (x + 0.044715 * x * x * x)))
    }
    private func moe(_ h: MLXArray, _ prefix: String) -> MLXArray {
        let r = softmax(matmul(h, p.t(prefix + "moe.gate.weight").T), axis: -1)
        let experts = newGELU(matmul(h, p.t(prefix + "moe.experts.weight").T)).reshaped(
            h.dim(0), -1, 64)
        return linear(newGELU((experts * r[0..., 0..., .newAxis]).sum(axis: 1)), prefix + "ffn_out")
    }
    private func positional(_ h: MLXArray, _ prefix: String) -> MLXArray {
        let w = p.t(prefix + "pos.dw.weight")
        let padding = w.dim(1) / 2
        let padded = concatenated([.zeros([padding, 64]), h, .zeros([padding, 64])])
        var acc = conv1d(padded[.newAxis], w[0..., 0..., .newAxis], groups: 64)[0]
        acc = bn(acc + p.t(prefix + "pos.dw.bias"), prefix + "pos.bn")
        return linear(gelu(acc), prefix + "pos.pw")
    }
    private func flatten(_ x: MLXArray) -> MLXArray {
        x[0].transposed(1, 0, 2).reshaped(x.dim(2), -1)
    }
    private func block(_ x: MLXArray, _ xf: MLXArray, depth: Int, sf: Int, st: Int, prefix: String)
        -> (MLXArray, MLXArray)
    {
        let time = x.dim(2)
        var y = x
        for r in 0 ..< depth {
            y = residual(y, prefix + "cnn.\(r).", r == 0 ? sf : 1, r == 0 ? st : 1)
        }
        if p.tables[prefix + "reduce.weight"] != nil {
            y = bn(linear(y, prefix + "reduce"), prefix + "reduce.bn")
        }
        let frequency = y.dim(1)
        let channels = y.dim(3)
        let x2 = flatten(y)
        var h = ln(linear(x2, prefix + "rproj"), prefix + "rproj.ln")
        h = h + positional(h, prefix)
        h = ln(h + attention(h, prefix), prefix + "ln1")
        h = ln(h + moe(h, prefix), prefix + "ln2")
        var e = linear(h, prefix + "eproj") + x2
        if st > 1 { e = repeated(e, count: st, axis: 0)[0 ..< time] }
        let out = e + xf
        return (out.reshaped(out.dim(0), frequency, channels).transposed(1, 0, 2)[.newAxis], out)
    }

    func embed(_ pcm: MLXArray) throws -> MLXArray {
        let wave = Self.resample(
            pcm[0 ..< min(pcm.size, 60 * 44100)], kernel: resampleKernel, orig: 441, new: 160,
            width: 17)
        let features = try fbank(wave.asArray(Float.self))
        var x = features.T[.newAxis, 0..., 0..., .newAxis]
        let w = p.t("fe.head.conv.weight").transposed(0, 2, 3, 1)
        x = bn(conv2d(x, w, padding: 1) + p.t("fe.head.conv.bias"), "fe.head.bn")
        var xf = flatten(x)
        var stages: [MLXArray] = []
        for (stage, count) in [2, 4, 5, 3].enumerated() {
            for b in 0 ..< count {
                let sf = stage > 0 && b == 0 ? 2 : 1
                let st = stage > 0 && b > 0 ? 2 : 1
                (x, xf) = block(
                    x, xf, depth: stage == 3 && b > 0 ? 2 : 3, sf: sf, st: st,
                    prefix: "fe.b\(stage + 1).\(b).")
                eval(x, xf)
            }
            stages.append(xf)
        }
        let dense = p.t("fe.dense_w")
        var fused = stages[0] * dense[0]
        for s in 1 ..< 4 { fused = fused + stages[s] * dense[s] }
        let alpha = softmax(linear(tanh(linear(fused, "pool.l1")), "pool.l2"), axis: 0)
        let mean = (alpha * fused).sum(axis: 0)
        let variance = (alpha * fused * fused).sum(axis: 0) - mean * mean
        let pooled = concatenated([mean, sqrt(maximum(variance, varianceFloor))])
        let embedding = matmul(bn(pooled, "bn"), p.t("linear.weight").T)
        eval(embedding)
        guard all(isFinite(embedding)).item(Bool.self) else {
            throw Zonos3Error.invalidInput("Non-finite speaker embedding")
        }
        return embedding
    }
}
