import CryptoKit
import Foundation
import Hub
import MLX
import MLXAudioCore
import MLXRandom
import Tokenizers

final class Zonos3TextTokenizer {
    let tokenizer: any Tokenizer
    init(directory: URL, config: Zonos3BackboneConfiguration) throws {
        let spec = config.textTokenizerSpec
        guard spec.kind == "superbpe", spec.offset == 4, spec.bos == 2, spec.eos == 3,
            config.textVocab == spec.vocabSize + 4,
            spec.path == URL(fileURLWithPath: spec.path).lastPathComponent
        else {
            throw Zonos3Error.invalidCheckpoint("Unsupported SuperBPE convention")
        }
        let raw = try Data(contentsOf: directory.appendingPathComponent(spec.path))
        let hash = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        guard hash == spec.sha256,
            let json = try JSONSerialization.jsonObject(with: raw) as? [NSString: Any],
            let model = json["model"] as? [String: Any],
            let vocab = model["vocab"] as? [String: Int],
            vocab.count == spec.vocabSize,
            json["padding"] is NSNull, json["truncation"] is NSNull
        else {
            throw Zonos3Error.invalidCheckpoint("SuperBPE artifact mismatch")
        }
        tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "PreTrainedTokenizer"], tokenizerData: Config(json)
        )
    }
    func encode(_ text: String) -> [Int] {
        [2] + tokenizer.encode(text: text, addSpecialTokens: false).map { $0 + 4 } + [3]
    }
}

enum Zonos3Generation {
    static func conditions(_ c: Zonos3BackboneConfiguration, options: Zonos3Options) throws -> [Int]
    {
        guard options.speed.isFinite, options.speed > 0, options.speakerCFGScale.isFinite,
            options.speakerCFGScale >= 0,
            options.minFrames >= 0, (0 ... 30).contains(options.silencePrefillFrames),
            options.headroomDB?.isFinite != false,
            options.conditionBuckets == nil || options.conditionValues == nil
        else {
            throw Zonos3Error.invalidInput("Invalid speed, guidance, conditions, or frame settings")
        }
        var values = options.conditionValues
        if options.speed != 1 {
            guard options.conditionBuckets == nil, values == nil else {
                throw Zonos3Error.invalidInput("Speed cannot be combined with explicit conditions")
            }
            values = ["speaking_rate": 15.193 * Double(options.speed)]
        }
        let keys = Set(values?.keys.map { $0 } ?? options.conditionBuckets?.keys.map { $0 } ?? [])
        guard keys.isSubset(of: Set(c.conditionNames)) else {
            throw Zonos3Error.invalidInput("Unknown acoustic condition")
        }
        return try zip(c.conditionNames, c.conditionBucketCounts).map { name, count in
            if let value = values?[name] {
                guard value.isFinite else {
                    throw Zonos3Error.invalidInput("Condition values must be finite")
                }
                return c.conditionEdges[name]!.filter { $0 <= value }.count
            }
            let bucket = options.conditionBuckets?[name] ?? count
            guard (0 ... count).contains(bucket) else {
                throw Zonos3Error.invalidInput("Invalid \(name) bucket")
            }
            return bucket
        }
    }
    static func delay(_ codes: MLXArray, pad: Int, flush: Bool = true) -> MLXArray {
        let length = codes.dim(0)
        let channels = codes.dim(1)
        let out = MLXArray.full(
            [length + (flush ? channels - 1 : 0), channels], values: MLXArray(pad), dtype: .int32)
        for cb in 0 ..< channels {
            let count = min(length, out.dim(0) - cb)
            if count > 0 { out[cb ..< (cb + count), cb] = codes[0 ..< count, cb] }
        }
        return out
    }
    static func aligned(_ rows: MLXArray, codebooks: Int, stop: Int? = nil, start: Int = 0)
        -> MLXArray
    {
        let length = min(max(0, rows.dim(0) - codebooks + 1), stop ?? Int.max)
        if length <= start { return .zeros([0, codebooks], dtype: .int32) }
        return stacked(
            (0 ..< codebooks).map { cb in rows[(start + cb) ..< (length + cb), cb] }, axis: -1)
    }
    static func sample(
        _ logits: MLXArray, config c: Zonos3BackboneConfiguration, step: Int,
        history: MLXArray, parameters p: GenerateParameters, options: Zonos3Options, hasPrefix: Bool
    ) -> MLXArray {
        let ids = MLXArray(0 ..< c.audioVocab)[.newAxis]
        var invalid = (ids .== c.audioPadId) .|| ((ids .>= c.codebookSize) .&& (ids .!= c.eoaId))
        let beforeMinimum =
            (options.minFrames + MLXArray(0 ..< c.nCodebooks))[0..., .newAxis] .> step
        invalid = logicalOr(invalid, logicalAnd(ids .== c.eoaId, beforeMinimum))
        var x = MLX.where(invalid, -Float.infinity, logits.asType(.float32))
        let window = min(max(0, p.repetitionContextSize), history.dim(0))
        let penalty = p.repetitionPenalty ?? 2.4
        if window > 0 && penalty > 1 && options.repetitionCodebooks != 0 {
            let recent = history[(history.dim(0) - window)...].T
            var valid = (recent .>= 0) .&& (recent .< c.codebookSize)
            if options.repetitionCodebooks >= 0 {
                let penalized =
                    MLXArray(0 ..< c.nCodebooks)[0..., .newAxis] .< options.repetitionCodebooks
                valid = logicalAnd(valid, penalized)
            }
            let counts = MLXArray.zeros(logits.shape, dtype: .int32).at[
                MLXArray(0 ..< c.nCodebooks)[0..., .newAxis],
                clip(recent, min: 0, max: c.audioVocab - 1)
            ].add(valid.asType(.int32))
            x = MLX.where(counts .> 0, MLX.where(x .> 0, x / penalty, x * penalty), x)
        }
        var token: MLXArray
        if p.temperature <= 0 {
            token = argMax(x, axis: -1)
        } else {
            x = x / p.temperature
            if p.topK > 0 && p.topK < c.audioVocab {
                let threshold = top(x, k: p.topK, axis: -1).min(axis: -1, keepDims: true)
                x = MLX.where(x .< threshold, -Float.infinity, x)
            }
            var probs = softmax(x, axis: -1)
            if p.topP > 0 && p.topP < 1 {
                let order = argSort(-probs, axis: -1)
                let ranked = takeAlong(probs, order, axis: -1)
                let selected = MLX.where(cumsum(ranked, axis: -1) - ranked .> p.topP, 0, ranked)
                probs = putAlong(.zeros(probs.shape), order, values: selected, axis: -1)
                probs = probs / maximum(probs.sum(axis: -1, keepDims: true), 1e-8)
            }
            if p.minP > 0 {
                probs = MLX.where(probs .< p.minP * probs.max(axis: -1, keepDims: true), 0, probs)
            }
            let greedy = ids .== argMax(x, axis: -1, keepDims: true)
            probs = MLX.where(probs.sum(axis: -1, keepDims: true) .<= 0, greedy, probs)
            token = MLXRandom.categorical(
                log(probs), key: p.seed.map { MLXRandom.key($0 &+ UInt64(step)) })
        }
        if !hasPrefix {
            token = MLX.where(MLXArray(0 ..< c.nCodebooks) .> step, c.audioPadId, token)
        }
        return token.asType(.int32)
    }
    static func output(_ audio: MLXArray, gain: Float) -> MLXArray {
        let scaled = audio.asType(.float32) * gain
        let knee = pow(Float(10), -1 / 20)
        let magnitude = abs(scaled)
        let limited = knee + (1 - knee) * tanh((magnitude - knee) / (1 - knee))
        return MLX.where(magnitude .> knee, sign(scaled) * limited, scaled)
    }
}
