# Qwen3-ASR & Qwen3-ForcedAligner

Speech-to-text and forced-alignment models based on Qwen3 ASR architecture.

## Supported Models

ASR (1.7B):

- [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
- [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)
- [mlx-community/Qwen3-ASR-1.7B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-6bit)
- [mlx-community/Qwen3-ASR-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-4bit)

ASR (0.6B):

- [mlx-community/Qwen3-ASR-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16)
- [mlx-community/Qwen3-ASR-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit)
- [mlx-community/Qwen3-ASR-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-6bit)
- [mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit)

Forced Alignment (0.6B):

- [mlx-community/Qwen3-ForcedAligner-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)
- [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)
- [mlx-community/Qwen3-ForcedAligner-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-6bit)
- [mlx-community/Qwen3-ForcedAligner-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-4bit)

## Swift Example (ASR)

```swift
import MLXAudioCore
import MLXAudioSTT

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
_ = sampleRate

let model = try await Qwen3ASRModel.fromPretrained("mlx-community/Qwen3-ASR-0.6B-4bit")
let output = model.generate(audio: audio, language: "English")
print(output.text)
```

## Streaming Example (ASR)

```swift
for try await event in model.generateStream(audio: audio, language: "English") {
    switch event {
    case .token(let token):
        print(token, terminator: "")
    case .result(let result):
        print("\nFinal text: \(result.text)")
    case .info:
        break
    }
}
```

## Live Streaming with Confucius4-R2T2

[Confucius4-R2T2](https://huggingface.co/mlx-community/Confucius4-R2T2-8bit) checkpoints are trained to re-prompt with their own committed text. Select the policy explicitly, because the checkpoint cannot be detected from its config.

```swift
let model = try await Qwen3ASRModel.fromPretrained("mlx-community/Confucius4-R2T2-8bit")
let session = StreamingInferenceSession(
    model: model,
    config: StreamingConfig(language: "English", policy: .stablePrefix)
)
Task {
    for await event in session.events {
        if case .confirmed(let text) = event { print(text) }
    }
}
session.feedAudio(samples: chunk)
session.stop()
```

A pass that falls behind real time merges the buffered chunks into one pass. Over 16 s the oldest 8 s of audio and the text committed from it leave the prompt, while `.confirmed` keeps the whole transcript.

The stable-prefix loop is adapted from [netease-youdao/Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) under the Apache 2.0 license, copyright The NetEase Youdao team and contributors. The encoder-block reuse follows [xocialize/qwen3-asr-mlx-swift](https://github.com/xocialize/qwen3-asr-mlx-swift) under the MIT License. Confucius4-R2T2 weights are released under the NetEase Model Use License.

## Swift Example (Forced Alignment)

```swift
import MLXAudioCore
import MLXAudioSTT

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
_ = sampleRate

let model = try await Qwen3ForcedAlignerModel.fromPretrained("mlx-community/Qwen3-ForcedAligner-0.6B-4bit")
let result = model.generate(
    audio: audio,
    text: "The transcript to align",
    language: "English"
)

for item in result.items {
    print("[\(item.startTime)s - \(item.endTime)s] \(item.text)")
}
```
