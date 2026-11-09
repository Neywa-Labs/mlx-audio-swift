// Copyright 2026 The NetEase Youdao team and contributors.
// SPDX-License-Identifier: Apache-2.0
// Stable-prefix streaming adapted from netease-youdao/Confucius4-R2T2 src/r2t2/r2t2_asr.py,
// src/example.py and src/ws_server.py at 72b81be.
// Swift shape of the text helpers follows xocialize/qwen3-asr-mlx-swift R2T2Stream.swift
// (MIT License, Copyright (c) 2026 xocialize).

import Foundation

protocol StablePrefixTextCodec: Sendable {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
}

struct StablePrefixPassRequest: Sendable {
    let window: [Float]
    let prefix: String
    let maxTokens: Int
}

struct StablePrefixPassResult: Sendable {
    let delta: String
    let confirmedText: String
    let provisionalText: String
}

struct StablePrefixState: Sendable {
    private struct LedgerEntry: Sendable {
        var samples: Int
        var text: String
    }

    let forcedLanguage: String?
    let codec: any StablePrefixTextCodec

    private(set) var window: [Float] = []
    private var ledger: [LedgerEntry] = []
    private var trimmedSamples = 0
    private(set) var language = ""
    private(set) var committedText = ""
    private var lastTokenIsChinese = false
    private var maxNewTokens = Self.firstPassBudget
    private var isFirstPass = true
    private var pendingConsumed = 0
    private var pendingPrefix = ""
    private var pendingPrefixText = ""

    private static let sampleRate = 16000
    private static let windowSamples = 16 * sampleRate
    private static let dropSamples = 8 * sampleRate
    private static let lookaheadSamples = 2560
    private static let chunkSamples = 2560
    private static let firstChunkSamples = chunkSamples + lookaheadSamples
    private static let unfixedTokens = 1
    private static let chunkBudget = Double(chunkSamples / 1280)
    private static let firstPassBudget = Double(firstChunkSamples / 1280)

    init(forcedLanguage: String?, codec: any StablePrefixTextCodec) {
        self.forcedLanguage = forcedLanguage
        self.codec = codec
    }

    var nextChunkSamples: Int { isFirstPass ? Self.firstChunkSamples : Self.chunkSamples }

    func samplesToConsume(buffered: Int) -> Int {
        let first = nextChunkSamples
        guard buffered >= first else { return 0 }
        return first + (buffered - first) / Self.chunkSamples * Self.chunkSamples
    }

    mutating func beginPass(samples: [Float], isFinal: Bool) -> StablePrefixPassRequest {
        window.append(contentsOf: samples)
        pendingConsumed = samples.count
        applyRollingWindow()

        let prefixText = ledger.map(\.text).joined()
        var prefix = prefixText
        if forcedLanguage == nil, !language.isEmpty {
            prefix = "language \(language)\(StablePrefixText.asrTag)" + prefixText
        }
        prefix = StablePrefixText.beforePipe(prefix)
        pendingPrefix = prefix
        pendingPrefixText = prefixText

        let chunks = max(1, (samples.count - (isFirstPass ? Self.lookaheadSamples : 0)) / Self.chunkSamples)
        let budget = Int(maxNewTokens) * chunks
        let windowSeconds = Double(window.count) / Double(Self.sampleRate)
        let finalBudget = max(24, Int((windowSeconds * 10).rounded(.up)))
        return StablePrefixPassRequest(window: window, prefix: prefix, maxTokens: isFinal ? finalBudget : budget)
    }

    mutating func completePass(generated: String, isFinal: Bool) -> StablePrefixPassResult {
        defer { isFirstPass = false }
        let prefixStripped = pendingPrefixText.trimmingCharacters(in: .whitespacesAndNewlines)
        let gen = StablePrefixText.normalizePunctByContext(generated)
            .replacingOccurrences(of: "\u{FFFD}", with: "")

        if isFinal {
            let raw = StablePrefixText.beforePipe(pendingPrefix + gen)
            let (parsedLanguage, parsedText) = StablePrefixText.parseASROutput(raw, forcedLanguage: forcedLanguage)
            let text = StablePrefixText.beforePipe(parsedText)
            language = parsedLanguage
            let finalStripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let newText = String(finalStripped.dropFirst(prefixStripped.count))
            ledger.append(LedgerEntry(samples: pendingConsumed, text: newText))
            let emitted = StablePrefixText.joining(newText, after: committedText, restarted: prefixStripped.isEmpty)
            committedText += emitted
            return StablePrefixPassResult(
                delta: emitted,
                confirmedText: committedText,
                provisionalText: ""
            )
        }

        var rawDecoded = pendingPrefix + gen
        var detectedLanguage: String?
        if forcedLanguage == nil {
            detectedLanguage = StablePrefixText.parseASROutput(rawDecoded, forcedLanguage: nil).language
        }
        if forcedLanguage == "Chinese" || detectedLanguage == "Chinese" {
            rawDecoded = StablePrefixText.stripChineseSpaces(rawDecoded)
        }
        let (parsedLanguage, text) = StablePrefixText.parseASROutput(rawDecoded, forcedLanguage: forcedLanguage)
        if let tagRange = rawDecoded.range(of: StablePrefixText.asrTag) {
            rawDecoded = String(rawDecoded[..<tagRange.lowerBound]) + StablePrefixText.asrTag + text
        } else {
            rawDecoded = text
        }
        rawDecoded = StablePrefixText.beforePipe(rawDecoded)

        var unfixed = Self.unfixedTokens
        var hasTag = false
        if let tagRange = rawDecoded.range(of: StablePrefixText.asrTag) {
            hasTag = true
            if rawDecoded[tagRange.upperBound...].isEmpty { unfixed = 0 }
        }
        var fixed = rollback(rawDecoded, unfixed: unfixed)
        if let tagRange = fixed.range(of: StablePrefixText.asrTag) {
            fixed = String(fixed[tagRange.upperBound...])
        }

        if !hasTag && forcedLanguage == nil {
            ledger.append(LedgerEntry(samples: pendingConsumed, text: ""))
            updateBudget(delta: "")
            return StablePrefixPassResult(
                delta: "",
                confirmedText: committedText,
                provisionalText: ""
            )
        }

        language = parsedLanguage
        let visibleText = StablePrefixText.beforePipe(text)
        let fixedStripped = fixed.trimmingCharacters(in: .whitespacesAndNewlines)
        var newText = fixedStripped.hasPrefix(prefixStripped) ? String(fixedStripped.dropFirst(prefixStripped.count)) : ""
        newText = StablePrefixText.beforePipe(newText)
        ledger.append(LedgerEntry(samples: pendingConsumed, text: newText))
        let emitted = StablePrefixText.joining(newText, after: committedText, restarted: prefixStripped.isEmpty)
        committedText += emitted
        updateBudget(delta: newText)

        let provisional = String(visibleText.trimmingCharacters(in: .whitespacesAndNewlines).dropFirst(fixedStripped.count))

        let hallucinated = StablePrefixText.detectHallucination(fixedStripped)
        if hallucinated { resetContext() }
        return StablePrefixPassResult(
            delta: emitted,
            confirmedText: committedText,
            provisionalText: hallucinated ? "" : provisional
        )
    }

    private mutating func applyRollingWindow() {
        while window.count > Self.windowSamples {
            window.removeFirst(Self.dropSamples)
            trimmedSamples += Self.dropSamples
            while let first = ledger.first, first.samples <= trimmedSamples {
                trimmedSamples -= first.samples
                ledger.removeFirst()
            }
        }
    }

    private mutating func updateBudget(delta: String) {
        if !delta.isEmpty {
            if let last = delta.last(where: { !$0.isWhitespace && !$0.isPunctuation && !$0.isSymbol }) {
                lastTokenIsChinese = last.unicodeScalars.contains(where: StablePrefixText.isCJK)
            }
            maxNewTokens = Self.chunkBudget
        } else if lastTokenIsChinese {
            maxNewTokens = Self.chunkBudget
        } else {
            maxNewTokens += 0.5
        }
        if lastTokenIsChinese { maxNewTokens *= 2 }
        maxNewTokens = min(2 * Self.chunkBudget, maxNewTokens)
    }

    private mutating func resetContext() {
        window.removeAll()
        ledger.removeAll()
        trimmedSamples = 0
        if forcedLanguage == nil { language = "" }
        lastTokenIsChinese = false
        maxNewTokens = Self.firstPassBudget
    }

    private func rollback(_ raw: String, unfixed: Int) -> String {
        let ids = codec.encode(raw)
        var k = unfixed
        while true {
            let end = max(0, ids.count - k)
            let decoded = end > 0 ? codec.decode(Array(ids[0..<end])) : ""
            if !decoded.contains("\u{FFFD}") { return decoded }
            if end == 0 { return "" }
            k += 1
        }
    }
}

enum StablePrefixText {
    static let asrTag = "<asr_text>"
    private static let en2zh: [Unicode.Scalar: Unicode.Scalar] = [
        ",": "\u{FF0C}", ".": "\u{3002}", "!": "\u{FF01}", "?": "\u{FF1F}", ";": "\u{FF1B}", ":": "\u{FF1A}",
        "(": "\u{FF08}", ")": "\u{FF09}",
    ]
    private static let zh2en: [Unicode.Scalar: Unicode.Scalar] = Dictionary(
        uniqueKeysWithValues: en2zh.map { ($0.value, $0.key) }
    )
    private static let allPunct: Set<Unicode.Scalar> = Set(en2zh.keys).union(Set(en2zh.values))
    private static let quotes: Set<Unicode.Scalar> = ["\"", "'"]

    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(scalar.value)
    }

    static func normalizePunctByContext(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var output = scalars
        for (index, scalar) in scalars.enumerated() where allPunct.contains(scalar) {
            var previous: Unicode.Scalar?
            var cursor = index - 1
            while cursor >= 0 {
                if !scalars[cursor].properties.isWhitespace {
                    previous = scalars[cursor]
                    break
                }
                cursor -= 1
            }
            guard let previous else { continue }
            if isCJK(previous) {
                output[index] = en2zh[scalar] ?? scalar
            } else if previous.isASCII,
                      previous.properties.isAlphabetic || (0x30...0x39).contains(previous.value) || quotes.contains(previous) {
                output[index] = zh2en[scalar] ?? scalar
            }
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: output)
        return String(view)
    }

    static func beforePipe(_ text: String) -> String {
        guard let pipe = text.firstIndex(of: "|") else { return text }
        return String(text[..<pipe])
    }

    static func stripChineseSpaces(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var output = String.UnicodeScalarView()
        var index = 0
        while index < scalars.count {
            let scalar = scalars[index]
            if scalar.properties.isWhitespace, let last = output.last, isCJK(last) {
                var cursor = index
                while cursor < scalars.count, scalars[cursor].properties.isWhitespace { cursor += 1 }
                if cursor < scalars.count, isCJK(scalars[cursor]) {
                    index = cursor
                    continue
                }
            }
            output.append(scalar)
            index += 1
        }
        return String(output)
    }

    static func parseASROutput(_ raw: String, forcedLanguage: String?) -> (language: String, text: String) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let forcedLanguage { return (forcedLanguage, text) }
        let parsed = Qwen3ASRModel.extractLanguage(from: text)
        let language = parsed.language.flatMap { $0.lowercased() == "none" ? nil : $0 } ?? ""
        return (language, parsed.text)
    }

    static func joining(_ piece: String, after emitted: String, restarted: Bool) -> String {
        guard restarted,
              let left = emitted.last, let right = piece.first,
              !left.isWhitespace, !right.isWhitespace, !right.isPunctuation,
              !left.unicodeScalars.contains(where: isCJK), !right.unicodeScalars.contains(where: isCJK)
        else { return piece }
        return " " + piece
    }

    static func detectHallucination(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let maxPattern = 50
        let tail = 256
        let punctuation = Set("\u{FF0C}\u{3002}\u{FF01}\u{FF1F}\u{3001}\u{FF1B}\u{FF1A},.!?;:~\u{2026}\u{00B7}\"'()\u{FF08}\u{FF09}\u{300A}\u{300B}\u{2014}-")
        let suffix = Array(text.suffix(tail))
        let isBlank: (ArraySlice<Character>) -> Bool = { pattern in
            pattern.allSatisfy { punctuation.contains($0) || $0 == " " || $0 == "\t" }
        }
        if endsInRepeats(suffix, lengths: 1...maxPattern, skipping: isBlank) { return true }
        let normalized = Array(String(suffix.filter { !punctuation.contains($0) && !$0.isWhitespace }).lowercased())
        return endsInRepeats(normalized, lengths: 3...maxPattern, skipping: { _ in false })
    }

    private static func endsInRepeats(
        _ text: [Character],
        lengths: ClosedRange<Int>,
        skipping skip: (ArraySlice<Character>) -> Bool
    ) -> Bool {
        let threshold = 5
        for length in lengths where text.count >= length * threshold {
            let pattern = text.suffix(length)
            if skip(pattern) { continue }
            let repeats = (1..<threshold).allSatisfy { index in
                text[(text.count - (index + 1) * length)..<(text.count - index * length)].elementsEqual(pattern)
            }
            if repeats { return true }
        }
        return false
    }
}
