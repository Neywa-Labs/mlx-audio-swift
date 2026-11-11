import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXAudioCore
@testable import MLXAudioTTS

private let mediumConfigJSON = #"""
{"sample_rate": 44100, "sample_size": 16777216, "model": {
  "diffusion": {"config": {"embed_dim": 1536, "depth": 24, "ff_kwargs": {"mult": 4.0},
    "attn_kwargs": {"differential": true}}},
  "pretransform": {"config": {"decoder": {"config": {"channels": 256, "c_mults": [6],
    "transformer_depths": [12], "sinusoidal_blocks": [8], "sliding_window": [1, 1]}}}}}}
"""#

private final class IndexDecoder: StableAudio3Decoder {
    var windows: [Int] = []

    override func callAsFunction(_ latents: MLXArray) -> MLXArray {
        windows.append(latents.dim(1))
        return repeated(latents[0..., 0..., ..<1], count: 16, axis: 1) * MLXArray.ones([1, 1, 512])
    }
}

@Suite("Stable Audio 3", .serialized)
struct StableAudio3Tests {
    private func expectClose(_ value: MLXArray, _ expected: MLXArray, tolerance: Float = 1e-5) {
        #expect(abs(value - expected).max().item(Float.self) <= tolerance)
    }

    @Test func routing() {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/stable-audio-3-small-sfx") == "stable_audio_3")
    }

    @Test func fourierFeaturesUseTwoPiAndCosBeforeSin() {
        let features = stableAudio3FourierFeatures(MLXArray([Float(0.25)]))[0]
        #expect(features.shape == [256])
        let expected = 2 * Double.pi * 0.5 * 0.25
        #expect(abs(features[0].item(Float.self) - Float(cos(expected))) < 1e-5)
        #expect(abs(features[128].item(Float.self) - Float(sin(expected))) < 1e-5)
        expectClose(stableAudio3FourierFeatures(MLXArray([Float(0)]))[0, ..<128], MLXArray.ones([128]))
    }

    @Test func schedulePinsItsEndpoints() {
        let schedule = stableAudio3SamplingSchedule(steps: 2)
        #expect(schedule[0].item(Float.self) == 1)
        #expect(schedule[2].item(Float.self) == 0)
        #expect(abs(schedule[1].item(Float.self) - Float(1 / (1 + exp(-2.1)))) < 1e-5)
    }

    @Test func pingPongAddsNoiseBetweenStepsOnly() {
        let start = MLXArray.ones([1, 3, 2])
        let key = MLXRandom.key(3)
        var seen: [Float] = []
        let out = stableAudio3PingPong(
            model: { x, t in
                seen.append(t[0].item(Float.self))
                return zeros(like: x)
            },
            start: start, sigmas: MLXArray([Float(1), 0.5, 0]), key: key)
        let noise = MLXRandom.normal(start.shape, dtype: start.dtype, key: MLXRandom.split(key: key).1)
        #expect(seen == [1, 0.5])
        expectClose(out, 0.5 * start + 0.5 * noise)
    }

    @Test func guidanceKeepsOnlyTheOrthogonalDifference() {
        let x = MLXArray([Float(1), 1]).reshaped([1, 2, 1])
        let sigma = MLXArray(Float(0.5)).reshaped([1, 1, 1])
        let conditional = zeros(like: x)
        let parallel = MLXArray([Float(2), 2]).reshaped([1, 2, 1])
        let orthogonal = MLXArray([Float(1), -1]).reshaped([1, 2, 1])
        expectClose(stableAudio3GuidedVelocity(
            x: x, sigma: sigma, conditional: conditional, unconditional: parallel, cfgScale: 4), conditional)
        expectClose(stableAudio3GuidedVelocity(
            x: x, sigma: sigma, conditional: conditional, unconditional: orthogonal, cfgScale: 2), -orthogonal)
    }

    @Test func latentLengthIsNaturalCeil() {
        let model = StableAudio3Model()
        #expect(model.latentLength(seconds: 0.05) == 1)
        #expect(model.latentLength(seconds: 30) == 323)
    }

    @Test func secondsConditioningRoundsUpToWholeSeconds() {
        let model = StableAudio3Model()
        expectClose(model.secondsEmbedding(0.4), model.secondsEmbedding(1), tolerance: 0)
        #expect(abs(model.secondsEmbedding(1.5) - model.secondsEmbedding(1)).max().item(Float.self) > 0)
    }

    @Test func invalidRequestsAreRejected() {
        let model = StableAudio3Model()
        #expect(throws: AudioGenerationError.self) { try model.generate(text: " ") }
        #expect(throws: AudioGenerationError.self) { try model.generate(text: "x", options: .init(duration: 0)) }
        #expect(throws: AudioGenerationError.self) { try model.generate(text: "x", options: .init(duration: 120.1)) }
        #expect(throws: AudioGenerationError.self) { try model.generate(text: "x", options: .init(steps: 0)) }
    }

    @Test(arguments: [3, 7, 12, 13, 30])
    func decodeProducesWholeLatents(length: Int) {
        let decoder = StableAudio3Decoder(config: .init(depth: 2))
        let latents = MLXRandom.normal([1, length, 256], key: MLXRandom.key(0))
        let audio = decoder.decode(latents)
        #expect(audio.shape == [1, length * 4096, 2])
        if length == 30 {
            let whole = decoder(latents).reshaped([1, length * 16, 2, 256]).transposed(0, 1, 3, 2)
            expectClose(audio, whole.reshaped([1, length * 4096, 2]))
        }
    }

    @Test func sanitizeRenamesAndDrops() {
        let weight = MLXArray.zeros([1])
        let source = [
            "model.model.transformer.memory_tokens",
            "model.model.transformer.layers.0.pre_norm.gamma",
            "model.model.transformer.rotary_pos_emb.inv_freq",
            "conditioner.conditioners.prompt.padding_embedding",
            "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
            "pretransform.model.bottleneck.running_std",
            "pretransform.model.bottleneck.bias",
            "pretransform.model.bottleneck.scaling_factor",
            "pretransform.model.encoder.layers.0.weight",
            "pretransform.model.decoder.layers.1.weight",
            "pretransform.model.decoder.layers.3.new_tokens",
            "pretransform.model.decoder.layers.3.transformers.0.pre_norm.gamma",
        ]
        let sanitized = StableAudio3Model.sanitize(Dictionary(uniqueKeysWithValues: source.map { ($0, weight) }))
        #expect(Set(sanitized.keys) == [
            "dit.transformer.memory_tokens",
            "dit.transformer.layers.0.pre_norm.weight",
            "padding_embedding",
            "seconds_proj.bias",
            "decoder.running_std",
            "decoder.project_in.weight",
            "decoder.new_tokens",
            "decoder.transformers.0.pre_norm.gamma",
        ])
    }

    private static let smallLayout: [String: [Int]] = [
        "dit.transformer.memory_tokens": [64, 1024],
        "dit.transformer.global_cond_embedder.2.weight": [6144, 1024],
        "dit.preprocess_conv.weight": [256, 1, 256],
        "dit.to_cond_embed.0.weight": [1024, 768],
        "dit.to_timestep_embed.0.weight": [1024, 256],
        "dit.transformer.layers.19.to_scale_shift_gate": [6144],
        "dit.transformer.layers.19.to_local_embed.0.weight": [1024, 257],
        "dit.transformer.layers.19.ff.ff.0.proj.weight": [8192, 1024],
        "dit.transformer.layers.19.self_attn.to_qkv.weight": [3072, 1024],
        "dit.transformer.layers.19.cross_attn.to_kv.weight": [2048, 1024],
        "padding_embedding": [768],
        "seconds_proj.weight": [768, 256],
        "decoder.running_std": [1],
        "decoder.project_in.weight": [768, 256],
        "decoder.new_tokens": [1, 1, 768],
        "decoder.mapping.weight": [512, 3, 768],
        "decoder.transformers.5.self_attn.to_qkv.weight": [3840, 768],
        "decoder.transformers.5.self_attn.q_norm.alpha": [1],
        "decoder.transformers.5.ff.ff.0.proj.weight": [4608, 768],
        "text_encoder.layers.11.self_attn.q_proj.weight": [768, 768],
        "text_encoder.layers.11.mlp.gate_proj.weight": [2048, 768],
    ]

    private static let mediumLayout: [String: [Int]] = [
        "dit.transformer.memory_tokens": [64, 1536],
        "dit.transformer.layers.23.self_attn.to_qkv.weight": [7680, 1536],
        "dit.transformer.layers.23.cross_attn.to_q.weight": [3072, 1536],
        "dit.transformer.layers.23.cross_attn.to_kv.weight": [4608, 1536],
        "dit.transformer.layers.23.ff.ff.0.proj.weight": [12288, 1536],
        "decoder.project_in.weight": [1536, 256],
        "decoder.new_tokens": [1, 1, 1536],
        "decoder.mapping.weight": [512, 1, 1536],
        "decoder.transformers.11.self_attn.to_qkv.weight": [7680, 1536],
        "decoder.transformers.11.ff.ff.0.proj.weight": [9216, 1536],
    ]

    private func mediumConfiguration() throws -> StableAudio3Configuration {
        try StableAudio3Configuration.decoded(from: Data(mediumConfigJSON.utf8))
    }

    private func mediumModel() throws -> StableAudio3Model {
        let configuration = try mediumConfiguration()
        return StableAudio3Model(
            sampleRate: configuration.sampleRate, sampleSize: configuration.sampleSize,
            dit: configuration.dit, decoder: configuration.decoder)
    }

    @Test(arguments: [false, true])
    func parametersMatchCheckpointLayout(medium: Bool) throws {
        let model = medium ? try mediumModel() : StableAudio3Model()
        if medium {
            #expect(model.maxDuration == 380)
        }
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        for (key, shape) in medium ? Self.mediumLayout : Self.smallLayout {
            #expect(shapes[key] == shape, "\(key)")
        }
    }

    @Test func lastDecoderBlocksGateValueBySineOfPiGate() throws {
        let decoder = StableAudio3Decoder(config: try mediumConfiguration().decoder)
        let gate = StableAudio3GatedProjection(dim: 2, inner: 1, sinusoidal: true)
        try gate.update(
            parameters: ModuleParameters.unflattened(["proj.weight": MLXArray.identity(2), "proj.bias": MLXArray.zeros([2])]),
            verify: [])
        let gated = decoder.transformers.map { ($0.ff.ff[0] as! StableAudio3GatedProjection).sinusoidal }
        let out = gate(MLXArray([Float(0.5), 0.25]).reshaped([1, 2]))
        #expect(gated == Array(repeating: false, count: 5) + Array(repeating: true, count: 7))
        #expect(abs(out[0, 0].item(Float.self) - Float(0.5 * sin(Double.pi * 0.25))) < 1e-5)
    }

    @Test func slidingWindowAttentionEqualsBandLimitedDenseAttention() {
        let length = 5 * 17
        let attention = StableAudio3DifferentialAttention(dim: 128, slidingWindow: true)
        let x = MLXRandom.normal([1, length, 128], key: MLXRandom.key(0))
        let parts = split(attention.toQKV(x), parts: 5, axis: -1).map(stableAudio3ToHeads)
        let positions = MLXArray(0 ..< length)
        let band = abs(positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)) .<= 17
        func dense(_ query: MLXArray, _ keys: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(
                queries: stableAudio3Rope(attention.qNorm(query)),
                keys: stableAudio3Rope(attention.kNorm(keys)),
                values: parts[2],
                scale: stableAudio3AttentionScale,
                mask: .array(band)
            )
        }
        expectClose(
            attention.attendWindowed(parts[0], parts[1], parts[2], parts[3], parts[4]),
            dense(parts[0], parts[1]) - dense(parts[3], parts[4]),
            tolerance: 1e-4)
    }

    @Test(arguments: [(77, Set([77])), (144, Set([144])), (400, Set([144]))])
    func slidingWindowDecodeUsesConfigWindowsAndFullShortDecode(length: Int, windows: Set<Int>) throws {
        let decoder = IndexDecoder(config: try mediumConfiguration().decoder)
        let latents = broadcast(
            MLXArray(0 ..< length).asType(.float32).reshaped([1, length, 1]), to: [1, length, 256])
        let audio = decoder.decode(latents)
        #expect(Set(decoder.windows) == windows)
        expectClose(audio[0, .stride(by: 4096), 0], MLXArray(0 ..< length).asType(.float32), tolerance: 0)
    }

    @Test func guidanceComputesInFp32AndReturnsInputDtype() {
        let x = MLXArray([Float(300), 300]).reshaped([1, 2, 1]).asType(.float16)
        let sigma = MLXArray(Float(0.5)).reshaped([1, 1, 1]).asType(.float16)
        let conditional = zeros(like: x)
        let parallel = MLXArray([Float(2), 2]).reshaped([1, 2, 1]).asType(.float16)
        let out = stableAudio3GuidedVelocity(
            x: x, sigma: sigma, conditional: conditional, unconditional: parallel, cfgScale: 4)
        #expect(out.dtype == .float16)
        expectClose(out.asType(.float32), conditional.asType(.float32), tolerance: 1e-2)
    }

    @Test func pingPongKeepsTheLatentDtype() {
        let start = MLXArray.ones([1, 3, 2], dtype: .float16)
        var seen: [DType] = []
        let out = stableAudio3PingPong(
            model: { x, _ in
                seen.append(x.dtype)
                return zeros(like: x)
            },
            start: start, sigmas: MLXArray([Float(1), 0.5, 0]), key: MLXRandom.key(3))
        #expect(seen == [.float16, .float16])
        #expect(out.dtype == .float16)
    }

    @Test(arguments: [false, true])
    func forwardShapes(medium: Bool) {
        let frames = medium ? 3 : 2
        let latents = MLXArray.zeros([1, frames, 256])
        let velocity = StableAudio3DiT(config: .init(depth: 1, differential: medium))(
            latents, t: MLXArray([Float(0.5)]), crossAttention: MLXArray.zeros([1, 257, 768]),
            globalCond: MLXArray.zeros([1, 768]))
        let decoderConfig: StableAudio3DecoderConfig =
            medium ? .init(depth: 2, sinusoidalBlocks: 1, convMapping: false, slidingWindow: true) : .init(depth: 2)
        let patches = StableAudio3Decoder(config: decoderConfig)(latents)
        eval(velocity, patches)
        #expect(velocity.shape == [1, frames, 256])
        #expect(patches.shape == [1, 16 * frames, 512])
    }
}
