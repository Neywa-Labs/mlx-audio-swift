import MLX
import MLXNN
import Testing

@testable import MLXAudioTTS

@Suite("Chatterbox generation tests")
struct ChatterboxGenerationTests {
    @Test func emotionOverrideAppliesToDefaultConditioning() {
        let defaults = MLXArray(Float(0.5))

        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: nil).item(Float.self) == 0.5)
        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: 0.9).item(Float.self) == 0.9)
    }

    @Test func turboInputKeepsEmbeddingDtypeWithFloat32SpeakerEmbedding() {
        var hp = T3Configuration.turbo
        hp.textTokensDictSize = 32
        hp.speechTokensDictSize = 16
        hp.startSpeechToken = 14
        hp.stopSpeechToken = 15
        hp.maxTextTokens = 16
        hp.maxSpeechTokens = 16
        hp.speakerEmbedSize = 8
        let gpt2 = GPT2BackboneConfig(
            activationFunction: "gelu_new", nCtx: 64, hiddenSize: hp.nChannels, nHead: 16, nLayer: 1,
            vocabSize: 32, layerNormEpsilon: 1e-5)
        let model = T3GPT2Model(hp, gpt2Config: gpt2)
        let fp16 = model.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) }
        model.update(parameters: ModuleParameters.unflattened(fp16))

        var cond = T3Cond(speakerEmb: MLXArray.ones([1, 8], dtype: .float32))
        let input = model.prepareInputEmbeddings(t3Cond: &cond, textTokens: MLXArray([Int32(1), 2, 3]))

        #expect(input.dtype == .float16)
    }
}
