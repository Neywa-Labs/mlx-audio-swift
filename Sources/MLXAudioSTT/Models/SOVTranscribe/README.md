# SOV-Transcribe-1

[Salesforce SOV-Transcribe-1](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1)
combines a Whisper large-v3 audio encoder with a Gemma 4 mixture-of-experts
decoder (26.6B parameters, 4.6B active). It transcribes English, French, German,
Spanish, Italian, Portuguese and Dutch, following a natural-language instruction.

Supply mono audio at 16 kHz, up to 30 minutes. The recording is encoded in
30-second windows that share one decoder context, so long recordings need no
splitting.

## Supported models

| Repository ID | Download size |
| --- | --- |
| [`nativ-community/SOV-Transcribe-1-4bit`](https://huggingface.co/nativ-community/SOV-Transcribe-1-4bit) | 15.6 GB |
| [`nativ-community/SOV-Transcribe-1-8bit`](https://huggingface.co/nativ-community/SOV-Transcribe-1-8bit) | 28.2 GB |
| [`salesforce-sovereign-intelligence/SOV-Transcribe-1`](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1) (BF16) | 53.3 GB |

The quantized checkpoints keep the audio encoder and projector in BF16 and the
expert routers in float32. Transcribing with the 4-bit checkpoint peaks at about 16 GB of memory.

## CLI

```bash
mlx-audio-swift-stt --model nativ-community/SOV-Transcribe-1-4bit --audio speech.wav --output-path transcript --format txt
```

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await SOVTranscribeModel.fromPretrained("nativ-community/SOV-Transcribe-1-4bit")
let (_, audio) = try loadAudioArray(from: URL(fileURLWithPath: "speech.wav"), sampleRate: 16000)

// The model card's instruction for a known language.
let result = model.generate(
    audio: audio,
    prompt: "Transcribe the German audio into text, ensuring all punctuation marks are included."
)
print(result.text)
```

Without a prompt the model uses the reference implementation's default
instruction. `STTGenerateParameters.language` appends "The spoken language is
…" to the instruction, and `hotwords` appends a contextual vocabulary. For an
unknown language, the model card suggests "Transcribe the audio verbatim in the
language's standard writing system, using punctuation and capitalization. Output
only the transcript."

`generateStream` yields text deltas once the audio is encoded.

The model and the Salesforce reference code are licensed under
[Apache 2.0](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1#license).
The Gemma 4 decoder is adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
(MIT), following the Python port in [mlx-audio](https://github.com/Blaizzy/mlx-audio).
