# ZONOS3-Mini

Native Swift port of [Zyphra's ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini),
using [mlx-community/ZONOS3-Mini](https://huggingface.co/mlx-community/ZONOS3-Mini)
directly. No model conversion is needed.

Supports 44.1 kHz mono synthesis, 27 built-in voices, incremental streaming,
reference-audio voice cloning without a transcript, speaker embeddings and
guidance, speed, and acoustic conditions.

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await Zonos3Model.fromPretrained("mlx-community/ZONOS3-Mini")
let audio = try await model.generate(text: "Hello from Swift!", voice: "jett")
print(model.voices)

for try await event in model.generateStream(text: "Welcome back.", voice: "imogen") {
    if case .audio(let chunk) = event {
        // Play or accumulate chunk. Concatenate chunks for the complete waveform.
        print(chunk.size)
    }
}
```

The same repository ID works with `TTS.loadModel(modelRepo:)` and the TTS CLI:

```bash
mlx-audio-swift-tts --model mlx-community/ZONOS3-Mini \
  --text "Hello from ZONOS3 Mini!" --voice jett
```

Pass `refAudio:` as a materialized, finite mono waveform at **44,100 Hz** to
clone a voice. Reference audio is capped at 60 seconds. Use `voice: "none"`
for synthesis without a speaker prompt. `extractSpeakerEmbedding(_:)` returns
a reusable 192-dimensional vector accepted by `speakerEmbedding:`. Reference
audio and an explicit embedding are mutually exclusive.

`Zonos3Options` exposes `speed`, `speakerCFGScale`, `conditionValues` (for
`loudness`, `fmax`, and `speaking_rate`) or `conditionBuckets`,
`silencePrefillFrames`, `minFrames`, `repetitionCodebooks`, `headroomDB`, and
`textNormalization`. Speed cannot be combined with explicit conditions.
English number normalization is enabled by default. Token budgets count
delayed audio rows at 15 Hz. Serialize generation calls on each instance.

Speech editing, accent adapters, hidden-state steering, and raw reference-code
encoding follow the Python port's scope and are not exposed.

## Validation

Run the focused tests with Xcode to build and bundle the Metal library:

```bash
xcodebuild -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:MLXAudioTests/Zonos3Tests test
```

Tests use synthetic Python/NumPy reference values and do not download weights.
To enable synthesis, streaming, and cloning checks against a downloaded snapshot,
set `TEST_RUNNER_MLXAUDIO_ZONOS3_MODEL_DIR` to its directory when invoking
`xcodebuild`. CPU fixtures use strict FP32 tolerances; the optional full-checkpoint
comparisons allow the small differences between Metal matrix kernels.

Model weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Backbone, codec, and speaker encoder: adapted from
[Zyphra's MIT-licensed runtime](https://github.com/Zyphra/ZONOS3-Mini/blob/6ff97cac0c684732de5f2bc6e3671aa783c15054/LICENSE).
The Gated DeltaNet kernel includes
[Apple's MIT-licensed mlx-lm code](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).
