import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

/// The streaming encoder cache must return the last `slidingWindow - 1 + n` rows of the
/// whole stream, across storage growth and compaction.
struct VoxtralRealtimeEncoderStreamKVCacheTests {
    @Test func appendsMatchASlidingWindowOverTheWholeStream() {
        let width = 3
        let slidingWindow = 300
        // Grows in the first round; later rounds compact, some onto overlapping rows.
        let appendSizes = [1, 100, 200, 7, 1, 270]
        var nextValue = 0
        func rows(_ count: Int) -> MLXArray {
            let values = (nextValue..<(nextValue + count * width)).map(Float.init)
            nextValue += count * width
            return MLXArray(values, [count, width])
        }

        let cache = VoxtralRealtimeEncoderStreamKVCache(slidingWindow: slidingWindow)
        var stream: MLXArray?
        var total = 0
        var compactions = 0
        for _ in 0..<5 {
            for n in appendSizes {
                let newKeys = rows(n)
                let offsetBefore = cache.positionOffset
                let window = cache.append(keys: newKeys, values: newKeys + 0.5)
                if cache.positionOffset != offsetBefore { compactions += 1 }
                total += n
                stream = stream.map { MLX.concatenated([$0, newKeys], axis: 0) } ?? newKeys

                let kept = min(total, slidingWindow - 1 + n)
                let reference = stream![(total - kept)...]
                #expect(window.positionOffset == total - kept)
                #expect(window.keys.shape == reference.shape)
                #expect(MLX.arrayEqual(window.keys, reference).item(Bool.self))
                #expect(MLX.arrayEqual(window.values, reference + 0.5).item(Bool.self))
                #expect(cache.positionOffset + cache.count == total)
            }
        }
        #expect(compactions >= 4)
        // A window, the rows left to pile up before a compaction, and the largest append.
        let block = VoxtralRealtimeEncoderStreamKVCache.growthBlock
        let most = slidingWindow - 1 + block - 1 + appendSizes.max()!
        #expect(cache.keys!.shape[0] <= ((most + block - 1) / block) * block)
    }
}
