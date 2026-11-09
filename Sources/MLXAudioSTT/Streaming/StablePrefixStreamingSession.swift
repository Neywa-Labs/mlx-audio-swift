// Copyright 2026 The NetEase Youdao team and contributors.
// SPDX-License-Identifier: Apache-2.0
// Stable-prefix streaming adapted from netease-youdao/Confucius4-R2T2 src/r2t2/r2t2_asr.py,
// src/example.py and src/ws_server.py at 72b81be.
// The encoder-block and KV reuse follows xocialize/qwen3-asr-mlx-swift R2T2Stream.swift
// (MIT License, Copyright (c) 2026 xocialize).

import Foundation
import MLX
import MLXAudioCore
import Tokenizers
import os

struct StablePrefixDecodeOutput: Sendable {
    var text: String
    var reusedTokens: Int = 0
    var generatedTokens: Int = 0
}

protocol StablePrefixDecoding: AnyObject {
    func decode(window: [Float], prefix: String, maxTokens: Int) -> StablePrefixDecodeOutput
}

final class StablePrefixStreamingSessionCore: @unchecked Sendable, StreamingInferenceSessionCore {
    private struct Shared: Sendable {
        var pending: [Float] = []
        var isActive = true
        var isDraining = false
        var minimumSamples: Int
        var continuation: AsyncStream<TranscriptionEvent>.Continuation?
        var drainTask: Task<Void, Never>?
        var stopTask: Task<Void, Never>?
    }

    let events: AsyncStream<TranscriptionEvent>

    private let shared: OSAllocatedUnfairLock<Shared>
    private let decoder: any StablePrefixDecoding
    private var policy: StablePrefixState
    private var passIndex = 0
    private var cumulativeConsumed = 0

    convenience init(model: Qwen3ASRModel, language: String?) {
        guard let tokenizer = model.tokenizer else {
            preconditionFailure("StablePrefixStreamingSessionCore requires a loaded tokenizer")
        }
        let codec = StablePrefixTokenizerCodec(tokenizer: tokenizer)
        let forcedLanguage = model.normalizeLanguageName(language)
        self.init(
            decoder: Qwen3StablePrefixDecoder(model: model, codec: codec, forcedLanguage: forcedLanguage),
            codec: codec,
            forcedLanguage: forcedLanguage
        )
    }

    init(
        decoder: any StablePrefixDecoding,
        codec: any StablePrefixTextCodec,
        forcedLanguage: String?
    ) {
        self.decoder = decoder
        let policy = StablePrefixState(forcedLanguage: forcedLanguage, codec: codec)
        self.policy = policy
        var continuation: AsyncStream<TranscriptionEvent>.Continuation!
        self.events = AsyncStream { continuation = $0 }
        self.shared = OSAllocatedUnfairLock(
            uncheckedState: Shared(minimumSamples: policy.nextChunkSamples, continuation: continuation)
        )
    }

    func feedAudio(samples: [Float]) {
        shared.withLock { state in
            guard state.isActive else { return }
            state.pending.append(contentsOf: samples)
            guard !state.isDraining, state.pending.count >= state.minimumSamples else { return }
            state.isDraining = true
            state.drainTask = Task.detached { [self] in
                drain()
            }
        }
    }

    private func drain() {
        while !Task.isCancelled {
            let needed = policy.nextChunkSamples
            let want = shared.withLock { state -> [Float]? in
                state.minimumSamples = needed
                let count = policy.samplesToConsume(buffered: state.pending.count)
                guard state.isActive, count > 0 else {
                    state.isDraining = false
                    return nil
                }
                let chunk = Array(state.pending.prefix(count))
                state.pending.removeFirst(count)
                return chunk
            }
            guard let chunk = want else { return }
            autoreleasepool { runPass(samples: chunk, isFinal: false) }
        }
        shared.withLock { $0.isDraining = false }
    }

    func stop() {
        shared.withLock { state in
            guard state.isActive else { return }
            state.isActive = false
            let inFlight = state.drainTask
            state.drainTask = nil
            state.stopTask = Task.detached { [self] in
                _ = await inFlight?.value
                finish()
            }
        }
    }

    private func finish() {
        if Task.isCancelled { return }
        let tail = shared.withLock { state -> [Float] in
            let tail = state.pending
            state.pending.removeAll()
            return tail
        }
        if !tail.isEmpty || !policy.window.isEmpty {
            autoreleasepool { runPass(samples: tail, isFinal: true) }
        }
        let continuation = shared.withLock { state -> AsyncStream<TranscriptionEvent>.Continuation? in
            let continuation = state.continuation
            state.continuation = nil
            state.stopTask = nil
            return continuation
        }
        continuation?.yield(.ended(fullText: policy.committedText))
        continuation?.finish()
    }

    func cancel() {
        shared.withLock { state in
            state.isActive = false
            state.drainTask?.cancel()
            state.drainTask = nil
            state.stopTask?.cancel()
            state.stopTask = nil
            state.continuation?.finish()
            state.continuation = nil
        }
    }

    private func runPass(samples: [Float], isFinal: Bool) {
        let started = Date()
        let request = policy.beginPass(samples: samples, isFinal: isFinal)
        let output = decoder.decode(window: request.window, prefix: request.prefix, maxTokens: request.maxTokens)
        if Task.isCancelled { return }
        let result = policy.completePass(generated: output.text, isFinal: isFinal)

        passIndex += 1
        cumulativeConsumed += samples.count
        let seconds = Date().timeIntervalSince(started)
        let consumedSeconds = Double(samples.count) / 16000
        let continuation = shared.withLock { $0.continuation }

        if !result.delta.isEmpty {
            continuation?.yield(.confirmed(text: result.confirmedText))
        }
        continuation?.yield(.displayUpdate(confirmedText: result.confirmedText, provisionalText: result.provisionalText))
        continuation?.yield(.stats(StreamingStats(
            encodedWindowCount: passIndex,
            totalAudioSeconds: Double(cumulativeConsumed) / 16000,
            tokensPerSecond: seconds > 0 ? Double(output.generatedTokens) / seconds : 0,
            realTimeFactor: consumedSeconds > 0 ? seconds / consumedSeconds : 0,
            peakMemoryGB: Double(Memory.peakMemory) / 1e9
        )))

        Memory.clearCache()
    }
}

struct StablePrefixTokenizerCodec: StablePrefixTextCodec, @unchecked Sendable {
    let tokenizer: any Tokenizers.Tokenizer

    func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }
}

final class Qwen3StablePrefixDecoder: StablePrefixDecoding {
    private let model: Qwen3ASRModel
    private let codec: any StablePrefixTextCodec
    private let forcedLanguage: String?
    private let headIds: [Int]
    private let tailText: String
    private var cache: [KVCache]
    private var stableLength = 0
    private var blockMel: [MLXArray] = []
    private var blockFeatures: [MLXArray] = []

    private static let eosTokenIds = [151645, 151643]

    init(model: Qwen3ASRModel, codec: any StablePrefixTextCodec, forcedLanguage: String?) {
        self.model = model
        self.codec = codec
        self.forcedLanguage = forcedLanguage
        self.headIds = codec.encode(model.audioPromptHead())
        self.tailText = model.audioPromptTail(language: forcedLanguage)
        self.cache = model.makeCache()
    }

    func decode(window: [Float], prefix: String, maxTokens: Int) -> StablePrefixDecodeOutput {
        var samples = window
        if samples.count < 1280 {
            samples.append(contentsOf: repeatElement(0, count: 1280 - samples.count))
        }
        let (features, _, _) = model.preprocessAudio(MLXArray(samples))
        let mel = features[0].transposed(1, 0)
        eval(mel)
        let frames = mel.dim(0)
        let blockFrames = model.audioTower.nWindowInfer
        let completeBlocks = frames / blockFrames

        var encoded: [MLXArray] = []
        encoded.reserveCapacity(completeBlocks + 1)
        var stableBlocks = 0
        var runIsStable = true
        for block in 0..<completeBlocks {
            let slice = mel[(block * blockFrames)..<((block + 1) * blockFrames)]
            if runIsStable, block < blockMel.count,
               MLX.all(blockMel[block] .== slice).item(Bool.self) {
                encoded.append(blockFeatures[block])
                stableBlocks += 1
            } else {
                runIsStable = false
                let blockEncoded = model.audioTower.encodeSingleWindow(slice)
                eval(blockEncoded)
                if block < blockMel.count {
                    blockMel[block] = slice
                    blockFeatures[block] = blockEncoded
                } else {
                    blockMel.append(slice)
                    blockFeatures.append(blockEncoded)
                }
                encoded.append(blockEncoded)
            }
        }
        if blockMel.count > completeBlocks {
            blockMel.removeLast(blockMel.count - completeBlocks)
            blockFeatures.removeLast(blockFeatures.count - completeBlocks)
        }
        if completeBlocks * blockFrames < frames {
            let tailEncoded = model.audioTower.encodeSingleWindow(mel[(completeBlocks * blockFrames)...])
            eval(tailEncoded)
            encoded.append(tailEncoded)
        }

        let audio = encoded.count == 1 ? encoded[0] : MLX.concatenated(encoded, axis: 0)
        let tailIds = codec.encode(tailText + prefix)

        let stableAudio = encoded.prefix(stableBlocks).reduce(0) { $0 + $1.dim(0) }
        var keep = min(stableLength, headIds.count + stableAudio)
        let offset = cache.first?.offset ?? 0
        if keep > 0, offset >= keep {
            for layer in cache { _ = layer.trim(layer.offset - keep) }
        } else {
            keep = 0
            cache = model.makeCache()
        }

        let headEmbeddings = model.model.embedTokens(MLXArray(headIds.map(Int32.init)))
        let tailEmbeddings = model.model.embedTokens(MLXArray(tailIds.map(Int32.init)))
        let full = MLX.concatenated(
            [headEmbeddings, audio.asType(headEmbeddings.dtype), tailEmbeddings],
            axis: 0
        )
        let inputEmbeddings = full[keep...].expandedDimensions(axis: 0)

        var logits = model(inputIds: MLXArray([Int32(0)]), inputEmbeddings: inputEmbeddings, cache: cache)
        eval(logits)

        var generated: [Int] = []
        while generated.count < maxTokens, !Task.isCancelled {
            let next = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)
            if Self.eosTokenIds.contains(next) { break }
            generated.append(next)
            if codec.decode([next]).contains("|") { break }
            logits = model(inputIds: MLXArray([Int32(next)]).expandedDimensions(axis: 0), cache: cache)
            eval(logits)
        }

        stableLength = headIds.count + encoded.prefix(completeBlocks).reduce(0) { $0 + $1.dim(0) }

        let text = codec.decode(generated)
        return StablePrefixDecodeOutput(text: text, reusedTokens: keep, generatedTokens: generated.count)
    }
}
