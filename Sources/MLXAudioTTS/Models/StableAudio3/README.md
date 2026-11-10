# Stable Audio 3

Stable Audio 3 small generates 44.1 kHz stereo audio from a text prompt: sound effects with the `sfx` model and music with the `music` model. Clips can be up to 120 seconds long. Only the small models are supported.

## Supported models

- [mlx-community/stable-audio-3-small-sfx](https://huggingface.co/mlx-community/stable-audio-3-small-sfx)
- [mlx-community/stable-audio-3-small-music](https://huggingface.co/mlx-community/stable-audio-3-small-music)

## Swift

```swift
import MLXAudioTTS

let model = try await StableAudio3Model.fromPretrained(
    "mlx-community/stable-audio-3-small-sfx"
)
let audio = try model.generate(
    text: "A dog barking next to a waterfall",
    options: StableAudio3GenerationOptions(duration: 10, seed: 7)
)
// Stereo samples, shape (frames, 2), at model.sampleRate (44,100 Hz).
```

`StableAudio3GenerationOptions` takes `duration` in seconds, `steps` (default 8), `seed`, `cfgScale` (default 1, guidance off) and `negativePrompt`, which is only used when `cfgScale` is not 1.

Loading through `TTS.loadModel` also works. That path has no duration option, so it always generates 30 seconds. A prompt does not need a voice or reference audio, and reference audio is rejected.

## License

The Swift code in this folder follows Stability AI's MIT-licensed `optimized/mlx` code, see [LICENSE](LICENSE). The weights are under the [Stability AI Community License](https://stability.ai/community-license-agreement) and the text encoder under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Read both before you use or redistribute the model.
