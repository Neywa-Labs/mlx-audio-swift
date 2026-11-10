// Copyright (c) 2026 Zyphra, Inc. Adapted under the MIT License.
import Foundation
import MLX
import MLXFast
import MLXNN

final class Zonos3LMState {
    var position = 0
    var kv: [Int: (MLXArray, MLXArray)] = [:]
    var recurrent: [Int: MLXArray] = [:]
    var convolution: [Int: MLXArray] = [:]
    var arrays: [MLXArray] {
        kv.values.flatMap { [$0.0, $0.1] } + Array(recurrent.values) + Array(convolution.values)
    }
}

func zonos3RMS(_ x: MLXArray, _ weight: MLXArray? = nil, _ eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: weight ?? .ones([x.dim(-1)]), eps: eps)
}

final class Zonos3Backbone {
    let config: Zonos3BackboneConfiguration
    let p: Zonos3TensorStore
    init(_ config: Zonos3BackboneConfiguration, params: Zonos3TensorStore) {
        self.config = config
        self.p = params
    }

    private func attention(_ x: MLXArray, layer: Int, state: Zonos3LMState) -> MLXArray {
        let c = config
        let prefix = "blk.\(layer)."
        let n = x.dim(0)
        let lo = state.position
        let need = lo + x.dim(0)
        func lin(_ name: String) -> MLXArray { p.lin(prefix + name + ".weight", x) }
        let gate = sigmoid(lin("attn_gate"))
        var q = lin("attn_q").reshaped(n, c.nHeads, c.headDim)
        var k = lin("attn_k").reshaped(n, c.nKvHeads, c.headDim)
        let v = lin("attn_v").reshaped(n, c.nKvHeads, c.headDim).transposed(1, 0, 2)
        q = zonos3RMS(q, nil, c.qkEps) * p.t(prefix + "attn_temp")[0..., .newAxis]
        k = zonos3RMS(k, nil, c.qkEps)
        q = MLXFast.RoPE(
            q.transposed(1, 0, 2), dimensions: c.headDim,
            traditional: true, base: c.ropeTheta, scale: 1, offset: lo)
        k = MLXFast.RoPE(
            k.transposed(1, 0, 2), dimensions: c.headDim,
            traditional: true, base: c.ropeTheta, scale: 1, offset: lo)
        var kc: MLXArray
        var vc: MLXArray
        if let old = state.kv[layer] {
            (kc, vc) = old
        } else {
            kc = .zeros([c.nKvHeads, 0, c.headDim])
            vc = .zeros(kc.shape)
        }
        if kc.dim(1) < need {
            let pad = MLXArray.zeros([
                c.nKvHeads, ((need + 255) / 256) * 256 - kc.dim(1), c.headDim,
            ])
            kc = concatenated([kc, pad], axis: 1)
            vc = concatenated([vc, pad], axis: 1)
        }
        // MLXArray is a reference type: copy wrappers before in-place indexing cached arrays.
        kc = kc.reshaped(kc.shape)
        vc = vc.reshaped(vc.shape)
        kc[0..., lo ..< need, 0...] = k
        vc[0..., lo ..< need, 0...] = v
        state.kv[layer] = (kc, vc)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode =
            n > 1
            ? .array(
                MLXArray(0 ..< need)[.newAxis, 0...] .<= (lo + MLXArray(0 ..< n))[0..., .newAxis])
            : .none
        var out = MLXFast.scaledDotProductAttention(
            queries: q[.newAxis],
            keys: kc[.newAxis, 0..., 0 ..< need, 0...],
            values: vc[.newAxis, 0..., 0 ..< need, 0...],
            scale: pow(Float(c.headDim), -0.5), mask: mask)[0]
        out = out.transposed(1, 0, 2) * gate[0..., 0..., .newAxis]
        return p.lin(prefix + "attn_output.weight", out.reshaped(n, -1))
    }

    private func gdn(_ x: MLXArray, layer: Int, state: Zonos3LMState) -> MLXArray {
        let c = config
        let prefix = "blk.\(layer)."
        let n = x.dim(0)
        let h = c.gdnHeads
        let d = c.gdnHeadDim
        let inner = h * d
        let kernel = c.gdnKernel
        let projection = p.lin(prefix + "gdn_in.weight", x)
        let qkv = projection[0..., 0 ..< (3 * inner)]
        let z = projection[0..., (3 * inner) ..< (4 * inner)].reshaped(n, h, d)
        let a = projection[0..., (4 * inner) ..< (4 * inner + h)]
        let b = projection[0..., (4 * inner + h) ..< (4 * inner + 2 * h)]
        let history = state.convolution[layer] ?? .zeros([3 * inner, kernel - 1])
        let sequence = concatenated([history.T, qkv])
        let w = p.t(prefix + "gdn_conv.weight")
        var conv = sequence[0 ..< n] * w[0..., 0]
        for j in 1 ..< kernel { conv = conv + sequence[j ..< (j + n)] * w[0..., j] }
        conv = silu(conv)
        state.convolution[layer] = sequence[(sequence.dim(0) - kernel + 1)...].T
        var q = conv[0..., 0 ..< inner].reshaped(n, h, d)
        var k = conv[0..., inner ..< (2 * inner)].reshaped(n, h, d)
        let v = conv[0..., (2 * inner)...].reshaped(n, h, d)
        q = q * rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
        k = k * rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)
        let decay = exp(
            -exp(p.t(prefix + "gdn_A_log"))
                * logAddExp(a + p.t(prefix + "gdn_dt_bias"), MLXArray(0)))
        let initial = state.recurrent[layer] ?? .zeros([1, h, d, d])
        let (y, final) = Zonos3DeltaKernel.callAsFunction(
            q: (q * pow(Float(d), -0.5))[.newAxis], k: k[.newAxis],
            v: v[.newAxis], decay: decay[.newAxis], beta: sigmoid(b)[.newAxis], state: initial)
        state.recurrent[layer] = final
        let normalized = zonos3RMS(y[0], p.t(prefix + "gdn_norm.weight"), 1e-5)
        return p.lin(prefix + "attn_output.weight", (normalized * silu(z)).reshaped(n, inner))
    }

    private func feedForward(_ x: MLXArray, layer: Int) -> MLXArray {
        let c = config
        let prefix = "blk.\(layer)."
        func lin(_ name: String, _ input: MLXArray) -> MLXArray {
            p.lin(prefix + name + ".weight", input)
        }
        if p.matrices[prefix + "ffn_up.weight"] != nil {
            return lin("ffn_down", lin("ffn_up", x) * silu(lin("ffn_gate", x)))
        }
        let logits = lin("router", zonos3RMS(x, p.t(prefix + "router_norm.weight"), c.normEps))
        let choices = argPartition(
            p.t(prefix + "router_bias") - logits, kth: c.moeTopk - 1, axis: -1)[
                0..., 0 ..< c.moeTopk]
        let weights = softmax(takeAlong(logits, choices, axis: -1), axis: -1)
        let latent = lin("ffn_latent_down", x)[0..., .newAxis, .newAxis, 0...]
        let gate = p.matrices[prefix + "ffn_gate_exps.weight"]!(latent, indices: choices)
        let up = p.matrices[prefix + "ffn_up_exps.weight"]!(latent, indices: choices)
        let down = p.matrices[prefix + "ffn_down_exps.weight"]!(silu(gate) * up, indices: choices)
        return lin(
            "ffn_latent_up",
            (down[0..., 0..., 0, 0...] * weights[0..., 0..., .newAxis]).sum(axis: 1))
    }

    func callAsFunction(
        _ ids: MLXArray, state: Zonos3LMState,
        speaker: MLXArray? = nil, conditions: [Int]? = nil
    ) throws -> MLXArray {
        let c = config
        let lo = state.position
        guard ids.ndim == 2, ids.dim(0) > 0, ids.dim(1) == c.nCodebooks + 1,
            ids.dim(0) <= c.maxSeqlen - lo
        else {
            throw Zonos3Error.invalidInput("Invalid prompt shape or context exceeded")
        }
        let n = ids.dim(0)
        var x = p.t("audio_embd.0.weight")[ids[0..., 0]]
        for cb in 1 ..< c.nCodebooks { x = x + p.t("audio_embd.\(cb).weight")[ids[0..., cb]] }
        x = x + p.t("text_embd.weight")[ids[0..., c.nCodebooks]]
        if lo == 0 {
            if let speaker {
                x[0] =
                    p.lin("speaker_proj.weight", speaker.reshaped(1, -1))[0]
                    + p.t("speaker_proj.bias")
            }
            let offset = speaker == nil ? 0 : 1
            if let conditions {
                for (i, bucket) in conditions.enumerated() {
                    x[offset + i] = p.t("cond_embd.\(i).weight")[bucket]
                }
            }
        }
        x = zonos3RMS(x, nil, c.normEps)
        let queries = p.t("attnres_q.weight")
        var bank: [MLXArray] = []
        var scores: [MLXArray] = []
        func route(_ source: MLXArray, _ query: Int) -> MLXArray {
            bank.append(source)
            scores.append(matmul(zonos3RMS(source, nil, c.normEps), queries.T))
            if bank.count == 1 { return source }
            let w = softmax(stacked(scores.map { $0[0..., query] }), axis: 0)
            return (stacked(bank) * w[0..., 0..., .newAxis]).sum(axis: 0)
        }
        var mixed = route(x, 0)
        for layer in 0 ..< c.nLayers {
            let prefix = "blk.\(layer)."
            if layer > 0 { mixed = route(x, 2 * layer) }
            let input = zonos3RMS(mixed, p.t(prefix + "attn_norm.weight"), c.normEps)
            let out =
                (layer + 1) % c.gdnPeriod != 0
                ? gdn(input, layer: layer, state: state)
                : attention(input, layer: layer, state: state)
            mixed = route(out, 2 * layer + 1)
            x = feedForward(
                zonos3RMS(mixed, p.t(prefix + "ffn_norm.weight"), c.normEps), layer: layer)
        }
        mixed = route(x, 2 * c.nLayers)
        let logits = p.lin("output.weight", zonos3RMS(mixed, p.t("output_norm.weight"), c.normEps))
            .reshaped(n, c.nCodebooks, c.audioVocab)
        state.position = lo + n
        return c.lossSoftcap * tanh(logits / c.lossSoftcap)
    }
}
