import Foundation
import MLX
import MLXAudioCore
import MLXNN

/// Whisper large-v3 encoder with Transformers' layer names and stored position embeddings.
final class SOVTranscribeAudioEncoder: Module {
    let config: SOVTranscribeAudioConfig
    let conv1: Conv1d
    let conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    let layers: [WhisperEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: SOVTranscribeAudioConfig) {
        self.config = config
        conv1 = Conv1d(
            inputChannels: config.numMelBins, outputChannels: config.dModel, kernelSize: 3,
            padding: 1)
        conv2 = Conv1d(
            inputChannels: config.dModel, outputChannels: config.dModel, kernelSize: 3, stride: 2,
            padding: 1)
        _embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions, dimensions: config.dModel)
        var whisper = WhisperConfig()
        whisper.dModel = config.dModel
        whisper.encoderAttentionHeads = config.encoderAttentionHeads
        whisper.encoderFfnDim = config.encoderFFNDim
        layers = (0 ..< config.encoderLayers).map { _ in WhisperEncoderLayer(config: whisper) }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    /// Number of mel frames in one encoder window.
    var framesPerWindow: Int { 2 * config.maxSourcePositions }

    /// Encodes `[windows, frames, mels]` log-mel features into `[windows, positions, dModel]`.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        precondition(
            features.dim(1) == framesPerWindow,
            "Expected \(framesPerWindow) mel frames per audio window")
        var x = features.asType(conv1.weight.dtype)
        x = gelu(conv2(gelu(conv1(x))))
        x = (x + embedPositions.weight).asType(x.dtype)
        for layer in layers { x = layer(x) }
        return layerNorm(x)
    }
}

/// Projects groups of `downsample_factor` encoder frames into the decoder's embedding space.
final class SOVTranscribeAudioProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: SOVTranscribeConfig) {
        let hidden = config.textConfig.hiddenSize
        _linear1.wrappedValue = Linear(
            config.audioConfig.dModel * config.downsampleFactor, hidden, bias: false)
        _linear2.wrappedValue = Linear(hidden, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

enum SOVTranscribeAudio {
    static let sampleRate = 16000
    static let windowSamples = 30 * sampleRate
    /// The model card's limit for one recording.
    static let maximumSamples = 1800 * sampleRate

    static func windowCount(samples: Int) -> Int {
        max(1, (samples + windowSamples - 1) / windowSamples)
    }

    /// Log-mel features `[windows, 3000, mels]`. Whisper normalizes against the
    /// maximum of the whole padded recording, so the split follows extraction.
    static func features(_ audio: MLXArray, numMelBins: Int) -> MLXArray {
        let windows = windowCount(samples: audio.dim(0))
        let padded = MLX.padded(
            audio.asType(.float32), widths: [.init((0, windows * windowSamples - audio.dim(0)))])
        let mel = WhisperAudio.logMelSpectrogram(padded, nMels: numMelBins)
        return mel.transposed(1, 0).reshaped(windows, -1, numMelBins)
    }
}
