import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

private struct CharacterCodec: StablePrefixTextCodec {
    func encode(_ text: String) -> [Int] { text.map { Int($0.unicodeScalars.first!.value) } }
    func decode(_ ids: [Int]) -> String { String(String.UnicodeScalarView(ids.compactMap(Unicode.Scalar.init))) }
}

private struct SplitAccentCodec: StablePrefixTextCodec {
    let accent = Int(Character("\u{00E9}").unicodeScalars.first!.value)
    func encode(_ text: String) -> [Int] {
        text.flatMap { character -> [Int] in
            let value = Int(character.unicodeScalars.first!.value)
            return value == accent ? [value, -value] : [value]
        }
    }
    func decode(_ ids: [Int]) -> String {
        var scalars = String.UnicodeScalarView()
        var index = 0
        while index < ids.count {
            if ids[index] == accent {
                if index + 1 < ids.count, ids[index + 1] == -accent {
                    scalars.append(Unicode.Scalar(UInt32(accent))!)
                    index += 2
                } else {
                    scalars.append("\u{FFFD}")
                    index += 1
                }
            } else {
                scalars.append(Unicode.Scalar(UInt32(ids[index]))!)
                index += 1
            }
        }
        return String(scalars)
    }
}

private final class ScriptedDecoder: StablePrefixDecoding, @unchecked Sendable {
    private let lock = NSLock()
    private var script: [String]
    private var recorded: [(prefix: String, maxTokens: Int, windowCount: Int)] = []
    var gate: DispatchSemaphore?

    init(_ script: [String]) { self.script = script }

    var requests: [(prefix: String, maxTokens: Int, windowCount: Int)] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func decode(window: [Float], prefix: String, maxTokens: Int) -> StablePrefixDecodeOutput {
        lock.lock()
        recorded.append((prefix, maxTokens, window.count))
        lock.unlock()
        gate?.wait()
        lock.lock()
        defer { lock.unlock() }
        let text = script.isEmpty ? "" : script.removeFirst()
        return StablePrefixDecodeOutput(text: text, generatedTokens: text.count)
    }

    func reset() {}
}

private func waitForPasses(_ decoder: ScriptedDecoder, _ count: Int) async {
    while decoder.requests.count < count {
        try? await Task.sleep(nanoseconds: 2_000_000)
    }
}

private func makeState(
    language: String? = "English",
    codec: any StablePrefixTextCodec = CharacterCodec()
) -> StablePrefixState {
    StablePrefixState(forcedLanguage: language, codec: codec)
}

private func step(
    _ state: inout StablePrefixState,
    generated: String,
    samples: Int = 2560,
    isFinal: Bool = false
) -> (request: StablePrefixPassRequest, result: StablePrefixPassResult) {
    let request = state.beginPass(samples: [Float](repeating: 0, count: samples), isFinal: isFinal)
    let result = state.completePass(generated: generated, isFinal: isFinal)
    return (request, result)
}

private func collect(_ events: AsyncStream<TranscriptionEvent>) async -> [TranscriptionEvent] {
    var collected: [TranscriptionEvent] = []
    for await event in events { collected.append(event) }
    return collected
}

private func endedText(_ events: [TranscriptionEvent]) -> String? {
    for event in events { if case .ended(let text) = event { return text } }
    return nil
}

private func confirmedTexts(_ events: [TranscriptionEvent]) -> [String] {
    events.compactMap { if case .confirmed(let text) = $0 { return text } else { return nil } }
}

@Suite(.serialized)
struct StablePrefixPolicyTests {
    @Test func lastTokenIsHeldBack() {
        var state = makeState()
        #expect(step(&state, generated: "abcd", samples: 5120).result.confirmedText == "abc")
        #expect(step(&state, generated: "defg").result.confirmedText == "abcdef")
    }

    @Test func nextPromptCarriesOnlyCommittedText() {
        var state = makeState()
        _ = step(&state, generated: "hello w", samples: 5120)
        let next = step(&state, generated: " world")
        #expect(next.request.prefix == "hello")
        #expect(next.result.confirmedText == "hello worl")
    }

    @Test func pipeIsNeverCommittedOrShown() {
        var state = makeState()
        let separate = step(&state, generated: "abc|def", samples: 5120).result
        #expect(separate.confirmedText == "ab")
        #expect(separate.provisionalText == "c")

        var merged = makeState()
        let result = step(&merged, generated: "abc |def", samples: 5120).result
        #expect(result.confirmedText == "abc")
        #expect(!result.provisionalText.contains("|"))

        var afterPrefix = makeState()
        _ = step(&afterPrefix, generated: "abcd", samples: 5120)
        #expect(step(&afterPrefix, generated: "de|gh").result.confirmedText == "abcd")

        var final = makeState()
        #expect(step(&final, generated: "abc|def", samples: 5120, isFinal: true).result.confirmedText == "abc")
    }

    @Test func splitCharacterIsHeldUntilComplete() {
        var state = makeState(codec: SplitAccentCodec())
        #expect(step(&state, generated: "caf\u{00E9}", samples: 5120).result.confirmedText == "caf")
        #expect(step(&state, generated: "\u{00E9} x").result.confirmedText == "caf\u{00E9}")
    }

    @Test func budgetFollowsTheAuthorsSchedule() {
        var state = makeState()
        #expect(step(&state, generated: "ab", samples: 5120).request.maxTokens == 4)
        #expect(step(&state, generated: "bc").request.maxTokens == 2)
        let stalled = (0..<6).map { _ in step(&state, generated: "").request.maxTokens }
        #expect(stalled == [2, 2, 3, 3, 4, 4])

        var chinese = makeState(language: "Chinese")
        _ = step(&chinese, generated: "\u{4F60}\u{597D}\u{5417}", samples: 5120)
        #expect(step(&chinese, generated: "\u{5417}\u{4E16}\u{754C}").request.maxTokens == 4)
        #expect(step(&chinese, generated: "").request.maxTokens == 4)
    }

    @Test func coalescedPassScalesBudget() {
        var state = makeState()
        _ = step(&state, generated: "ab", samples: 5120)
        #expect(step(&state, generated: "c", samples: 2560 * 3).request.maxTokens == 6)
    }

    @Test func autoLanguageHidesMetadataUntilTagArrives() {
        var state = makeState(language: nil)
        let noTag = step(&state, generated: "langua", samples: 5120).result
        #expect(noTag.confirmedText.isEmpty)
        #expect(noTag.provisionalText.isEmpty)
        #expect(state.language.isEmpty)

        let tagged = step(&state, generated: "language English<asr_text>he").result
        #expect(tagged.confirmedText == "h")
        #expect(state.language == "English")

        let next = step(&state, generated: "ello")
        #expect(next.request.prefix == "language English<asr_text>h")
        #expect(next.result.confirmedText == "hell")
    }

    @Test func rollingWindowDropsTheAuthorsEntryCountsAtDefaultChunk() {
        var state = makeState()
        for index in 0..<99 {
            _ = step(&state, generated: "[\(index)]#", samples: index == 0 ? 5120 : 2560)
        }
        #expect(state.window.count == 256000)
        let firstDrop = step(&state, generated: "[99]#").request
        #expect(firstDrop.window.count == 256000 + 2560 - 128000)
        #expect(!firstDrop.prefix.contains("[48]"))
        #expect(firstDrop.prefix.hasPrefix("[49]"))

        for index in 100..<149 { _ = step(&state, generated: "[\(index)]#") }
        #expect(state.window.count == 256000)
        let secondDrop = step(&state, generated: "[149]#").request
        #expect(secondDrop.prefix.hasPrefix("[99]"))
    }

    @Test func entryStraddlingTheDropBoundaryIsKept() {
        var state = makeState()
        _ = step(&state, generated: "[a]#", samples: 120_000)
        _ = step(&state, generated: "[b]#", samples: 40_000)
        _ = step(&state, generated: "[c]#", samples: 90_000)
        let request = step(&state, generated: "[d]#", samples: 10_000).request
        #expect(request.window.count == 260_000 - 128_000)
        #expect(request.prefix == "[b][c]")
    }

    @Test func restartAfterEmptyPrefixGetsASeparator() {
        var state = makeState()
        _ = step(&state, generated: "lakes#", samples: 5120)
        for _ in 0..<110 { _ = step(&state, generated: "", samples: 2560) }
        #expect(state.committedText == "lakes")
        let restarted = step(&state, generated: "On August", samples: 2560).result
        #expect(restarted.confirmedText == "lakes On Augus")
    }

    @Test func hallucinationClearsContextButKeepsCommittedText() {
        var state = makeState()
        let result = step(&state, generated: "okay okay okay okay okay okay x", samples: 5120).result
        #expect(state.window.isEmpty)
        #expect(result.confirmedText == "okay okay okay okay okay okay")
        #expect(step(&state, generated: "hello", samples: 5120).request.prefix.isEmpty)
    }

    @Test func guardDoesNotFireAgainAfterClearingContext() {
        var state = makeState()
        _ = step(&state, generated: "okay okay okay okay okay okay x", samples: 5120)
        _ = step(&state, generated: "", samples: 5120)
        #expect(!state.window.isEmpty)
    }

    @Test func passLongerThanTheTrimStillLeavesThePrompt() {
        var state = makeState()
        _ = step(&state, generated: "[big]#", samples: 160_000)
        _ = step(&state, generated: "[a]#", samples: 128_000)
        let request = step(&state, generated: "[b]#", samples: 128_000).request
        #expect(request.prefix.hasPrefix("[a]"))
    }

    @Test func ordinaryRepetitionDoesNotTriggerTheGuard() {
        #expect(!StablePrefixText.detectHallucination("the cat sat on the mat and the cat sat down"))
        #expect(StablePrefixText.detectHallucination("ha ha ha ha ha ha"))
    }

    @Test func finalPassCommitsTailWithoutHoldingTokens() {
        var state = makeState()
        _ = step(&state, generated: "abcd", samples: 5120)
        let final = step(&state, generated: "def", isFinal: true)
        #expect(final.result.confirmedText == "abcdef")
        #expect(final.result.delta == "def")
        #expect(final.result.provisionalText.isEmpty)
        #expect(final.request.maxTokens >= 24)
    }

    // Expected values come from running _normalize_punct_by_context in r2t2_asr.py at 72b81be.
    @Test(arguments: [
        ("hello, world.", "hello, world."),
        ("\u{4F60}\u{597D},\u{4E16}\u{754C}.", "\u{4F60}\u{597D}\u{FF0C}\u{4E16}\u{754C}\u{3002}"),
        ("abc\u{FF0C}def\u{3002}", "abc,def."),
        ("\u{4F60}\u{597D} , \u{4E16}", "\u{4F60}\u{597D} \u{FF0C} \u{4E16}"),
        (",leading", ",leading"),
        ("(\u{4F60})", "(\u{4F60}\u{FF09}"),
        ("\u{4F60}\u{597D}\u{3002}abc,def", "\u{4F60}\u{597D}\u{3002}abc,def"),
    ])
    func punctuationNormalizationMatchesAuthors(input: String, expected: String) {
        #expect(StablePrefixText.normalizePunctByContext(input) == expected)
    }
}

@Suite(.serialized)
struct StablePrefixSessionCoreTests {
    private func makeCore(_ decoder: ScriptedDecoder) -> StablePrefixStreamingSessionCore {
        StablePrefixStreamingSessionCore(
            decoder: decoder,
            codec: CharacterCodec(),
            forcedLanguage: "English"
        )
    }

    @Test(arguments: [100, 10000])
    func everySampleIsConsumedOnceInWholeChunks(feedSize: Int) async {
        let decoder = ScriptedDecoder([])
        let core = makeCore(decoder)
        let total = 5120 + 2560 * 5 + 100
        var fed = 0
        while fed < total {
            let count = min(feedSize, total - fed)
            core.feedAudio(samples: [Float](repeating: 0, count: count))
            fed += count
        }
        core.stop()
        _ = await collect(core.events)
        let windows = decoder.requests.map(\.windowCount)
        #expect(windows.last == total)
        for window in windows.dropLast() {
            #expect(window >= 5120 && (window - 5120) % 2560 == 0)
        }
    }

    @Test func backlogIsCoalescedIntoOnePass() async {
        let decoder = ScriptedDecoder([])
        let gate = DispatchSemaphore(value: 0)
        decoder.gate = gate
        let core = makeCore(decoder)
        core.feedAudio(samples: [Float](repeating: 0, count: 5120))
        await waitForPasses(decoder, 1)
        core.feedAudio(samples: [Float](repeating: 0, count: 2560 * 3))
        gate.signal()
        await waitForPasses(decoder, 2)
        core.stop()
        gate.signal()
        gate.signal()
        _ = await collect(core.events)
        #expect(decoder.requests.map(\.windowCount) == [5120, 12800, 12800])
    }

    @Test func stopOnExactBoundaryStillRunsFinalPass() async {
        let decoder = ScriptedDecoder(["abcd", "def"])
        let core = makeCore(decoder)
        core.feedAudio(samples: [Float](repeating: 0, count: 5120))
        await waitForPasses(decoder, 1)
        core.stop()
        let events = await collect(core.events)
        #expect(decoder.requests.count == 2)
        #expect(endedText(events) == "abcdef")
        #expect(confirmedTexts(events) == ["abc", "abcdef"])
    }

    @Test func stopOnEmptySessionEndsWithoutDecoding() async {
        let decoder = ScriptedDecoder([])
        let core = makeCore(decoder)
        core.stop()
        let events = await collect(core.events)
        #expect(decoder.requests.isEmpty)
        #expect(endedText(events) == "")
    }

    @Test func stopDuringInFlightPassWaitsAndFinishes() async {
        let decoder = ScriptedDecoder(["abcd", "def"])
        let gate = DispatchSemaphore(value: 0)
        decoder.gate = gate
        let core = makeCore(decoder)
        core.feedAudio(samples: [Float](repeating: 0, count: 5120))
        await waitForPasses(decoder, 1)
        core.stop()
        gate.signal()
        gate.signal()
        let events = await collect(core.events)
        #expect(endedText(events) == "abcdef")
    }

    @Test func cancelFinishesEventStreamWithoutFinalText() async {
        let decoder = ScriptedDecoder(["abcd"])
        let core = makeCore(decoder)
        core.feedAudio(samples: [Float](repeating: 0, count: 5120))
        core.cancel()
        let events = await collect(core.events)
        #expect(endedText(events) == nil)
    }
}

@Suite(.serialized)
struct Qwen3StablePrefixDecoderTests {
    private func makeModel() -> Qwen3ASRModel {
        var config = Qwen3ASRConfig(
            audioConfig: Qwen3AudioEncoderConfig(
                numMelBins: 16,
                encoderLayers: 1,
                encoderAttentionHeads: 2,
                encoderFfnDim: 32,
                dModel: 16,
                outputDim: 16,
                downsampleHiddenSize: 4
            ),
            textConfig: Qwen3TextConfig(
                vocabSize: 64,
                hiddenSize: 16,
                intermediateSize: 32,
                numHiddenLayers: 1,
                numAttentionHeads: 2,
                numKeyValueHeads: 1,
                headDim: 8
            ),
            audioTokenId: 2,
            audioStartTokenId: 1,
            audioEndTokenId: 3,
            classifyNum: 4
        )
        config.supportLanguages = ["English"]
        let model = Qwen3ASRModel(config)
        eval(model)
        return model
    }

    private struct SmallCodec: StablePrefixTextCodec {
        func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) % 60 + 4 } }
        func decode(_ ids: [Int]) -> String { String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0 + 96)) })) }
    }

    private func samples(_ count: Int) -> [Float] {
        (0..<count).map { sin(Float($0) * 0.031) * (Float($0 % 977) / 977.0) }
    }

    @Test func reuseMatchesFullReencodeAcrossBlockBoundary() {
        let model = makeModel()
        let reused = Qwen3StablePrefixDecoder(model: model, codec: SmallCodec(), forcedLanguage: "English")
        var audio = samples(16000 * 10)
        audio[1600] = 1
        var sawReuse = false
        for end in stride(from: 120_000, through: 160_000, by: 2560) {
            let window = Array(audio[0..<end])
            let a = reused.decode(window: window, prefix: "abc", maxTokens: 4)
            let fresh = Qwen3StablePrefixDecoder(model: model, codec: SmallCodec(), forcedLanguage: "English")
            let b = fresh.decode(window: window, prefix: "abc", maxTokens: 4)
            #expect(a.text == b.text)
            sawReuse = sawReuse || a.reusedTokens > 0
        }
        #expect(sawReuse)
    }

    @Test func reuseIsDroppedWhenBlockMelChanges() {
        let model = makeModel()
        let decoder = Qwen3StablePrefixDecoder(model: model, codec: SmallCodec(), forcedLanguage: "English")
        var audio = samples(16000 * 9)
        audio[1600] = 1
        _ = decoder.decode(window: audio, prefix: "", maxTokens: 2)
        let extended = audio + samples(2560)
        let reused = decoder.decode(window: extended, prefix: "", maxTokens: 2)
        #expect(reused.reusedTokens > 0)
        var changed = extended
        changed[1600] = 0
        changed[3200] = 0.2
        let afterChange = decoder.decode(window: changed, prefix: "", maxTokens: 2)
        #expect(afterChange.reusedTokens < reused.reusedTokens)
    }

    @Test func shortInputIsPaddedBeforeDecoding() {
        let model = makeModel()
        let decoder = Qwen3StablePrefixDecoder(model: model, codec: SmallCodec(), forcedLanguage: "English")
        let output = decoder.decode(window: samples(300), prefix: "", maxTokens: 2)
        #expect(output.generatedTokens > 0)
    }
}
