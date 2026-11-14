import Foundation

public enum PocketTTSTextUtils {
    public static func prepareTextPrompt(_ text: String) throws -> (String, Int) {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty {
            throw NSError(domain: "PocketTTSTextUtils", code: 1, userInfo: [NSLocalizedDescriptionKey: "Text prompt cannot be empty"])
        }
        t = t.replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "  ", with: " ")

        let wordCount = t.split(separator: " ").count
        let framesAfterEosGuess = wordCount <= 4 ? 3 : 1

        if let first = t.first, !first.isUppercase {
            t = first.uppercased() + t.dropFirst()
        }
        if let last = t.last, last.isLetter || last.isNumber {
            t += "."
        }
        if t.split(separator: " ").count < 5 {
            t = String(repeating: " ", count: 8) + t
        }
        return (t, framesAfterEosGuess)
    }

    public static func splitIntoBestSentences(
        _ tokenizer: PocketTTSSentencePieceTokenizer,
        _ textToGenerate: String
    ) throws -> [String] {
        let (prepared, _) = try prepareTextPrompt(textToGenerate)
        let cleaned = prepared.trimmingCharacters(in: .whitespaces)

        let tokens = tokenizer.encode(cleaned)
        let endTokens = tokenizer.encode(".!...?")
        guard !tokens.isEmpty else { return [cleaned] }

        var endOfSentenceIndices: [Int] = [0]
        var previousWasEnd = false
        for (idx, tok) in tokens.enumerated() {
            if endTokens.contains(tok) {
                previousWasEnd = true
            } else {
                if previousWasEnd && !isInnerPunctuation(tokens, idx, tokenizer) {
                    endOfSentenceIndices.append(idx)
                }
                previousWasEnd = false
            }
        }
        endOfSentenceIndices.append(tokens.count)

        var tokenChunks: [(count: Int, text: String)] = []
        for i in 0..<(endOfSentenceIndices.count - 1) {
            let start = endOfSentenceIndices[i]
            let end = endOfSentenceIndices[i + 1]
            let slice = Array(tokens[start..<end])
            let text = tokenizer.decode(slice)
            tokenChunks.append((end - start, text))
        }

        let maxNbTokensInChunk = 50
        var chunks: [String] = []
        var current = ""
        var currentCount = 0
        for (count, text) in tokenChunks {
            if current.isEmpty {
                current = text
                currentCount = count
                continue
            }
            if currentCount + count > maxNbTokensInChunk {
                chunks.append(current.trimmingCharacters(in: .whitespaces))
                current = text
                currentCount = count
            } else {
                current += " " + text
                currentCount += count
            }
        }
        if !current.isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespaces))
        }
        return chunks
    }

    // Abbreviations that always precede a name or an example, never end a sentence.
    static let titles: Set<String> = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "vs.", "e.g.", "i.e."]

    /// Returns true when the punctuation right before `start` does not end a sentence: it is
    /// followed by a letter or digit without a space ("3.14", "4 p.m.", "www.kyutai.org",
    /// "jo@icloud.com"), or it ends a title that precedes a name ("Dr. Miller", "e.g. this"),
    /// as in pocket-tts.
    ///
    /// Only the tokens next to the boundary are decoded. Decoding the period together with the
    /// next token keeps the space between them: in "I counted to 3. 14 people were there." the
    /// period ends a sentence.
    static func isInnerPunctuation(
        _ tokens: [Int],
        _ start: Int,
        _ tokenizer: PocketTTSSentencePieceTokenizer
    ) -> Bool {
        let after = Array(tokenizer.decode(Array(tokens[(start - 1)...start])))
        if after.count >= 2, after[1].isLetter || after[1].isNumber {
            return true
        }
        let windowStart = max(0, start - 6)
        let words = tokenizer.decode(Array(tokens[windowStart..<start])).split(whereSeparator: \.isWhitespace)
        guard let last = words.last else { return false }
        // The first word of the window may be cut, unless the window starts the text.
        return titles.contains(String(last)) && (words.count > 1 || windowStart == 0)
    }
}
