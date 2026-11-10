// Adapted from Zyphra's MIT-licensed ZONOS3-Mini runtime.
import Foundation
import MLX

/// The published artifact uses encoded names and already-packed affine group-32 matrices.
/// Keep these tensors in their native layout; no conversion or requantization is needed.
final class Zonos3TensorStore {
    let tables: [String: MLXArray]
    let matrices: [String: Zonos3Matrix]

    init(
        specs: [String: Zonos3TensorSpecification], weights: [String: MLXArray], prefix: String = ""
    ) throws {
        var tables: [String: MLXArray] = [:]
        var matrices: [String: Zonos3Matrix] = [:]
        var expected = Set<String>()
        for (name, spec) in specs {
            guard !name.contains("__"), !spec.shape.isEmpty, spec.shape.allSatisfy({ $0 > 0 })
            else {
                throw Zonos3Error.invalidCheckpoint("Invalid tensor specification: \(name)")
            }
            let encoded = name.replacingOccurrences(of: ".", with: "__")
            func tensor(_ key: String, shape: [Int], dtype: DType) throws -> MLXArray {
                let full = prefix + key
                expected.insert(full)
                guard let value = weights[full], value.shape == shape, value.dtype == dtype else {
                    throw Zonos3Error.invalidCheckpoint("Missing or incompatible tensor: \(full)")
                }
                return value
            }
            let dtype: DType
            switch spec.dtype {
            case "uint32": dtype = .uint32
            case "float32": dtype = .float32
            case "float16": dtype = .float16
            case "bfloat16": dtype = .bfloat16
            default: throw Zonos3Error.invalidCheckpoint("Unsupported dtype: \(spec.dtype)")
            }
            if spec.matrix {
                let key = "matrices." + encoded
                if let bits = spec.bits {
                    guard [4, 8].contains(bits), spec.shape.count >= 2, spec.shape.last! % 32 == 0
                    else {
                        throw Zonos3Error.invalidCheckpoint("Unsupported packed matrix: \(name)")
                    }
                    var packed = spec.shape
                    var groups = spec.shape
                    packed[packed.count - 1] = packed.last! * bits / 32
                    groups[groups.count - 1] /= 32
                    matrices[name] = try Zonos3Matrix(
                        weight: tensor(key + ".weight", shape: packed, dtype: .uint32),
                        scales: tensor(key + ".scales", shape: groups, dtype: .float32),
                        biases: tensor(key + ".biases", shape: groups, dtype: .float32), bits: bits)
                } else {
                    matrices[name] = try Zonos3Matrix(
                        weight: tensor(key + ".weight", shape: spec.shape, dtype: dtype))
                }
            } else {
                tables[name] = try tensor("tables." + encoded, shape: spec.shape, dtype: dtype)
            }
        }
        guard Set(weights.keys) == expected else {
            throw Zonos3Error.invalidCheckpoint(
                "Unexpected checkpoint tensors: \(Set(weights.keys).subtracting(expected).sorted())"
            )
        }
        self.tables = tables
        self.matrices = matrices
    }

    func t(_ name: String) -> MLXArray { tables[name]! }
    func lin(_ name: String, _ x: MLXArray, bias: String? = nil) -> MLXArray {
        let y = matrices[name]!(x)
        return bias.map { y + t($0) } ?? y
    }
    var arrays: [MLXArray] { Array(tables.values) + matrices.values.flatMap(\.arrays) }
}

struct Zonos3Matrix {
    let weight: MLXArray
    var scales: MLXArray? = nil
    var biases: MLXArray? = nil
    var bits: Int? = nil
    var arrays: [MLXArray] { [weight] + [scales, biases].compactMap { $0 } }

    func callAsFunction(_ x: MLXArray, indices: MLXArray? = nil) -> MLXArray {
        if let bits, let scales {
            if let indices {
                return gatherQuantizedMM(
                    x, weight, scales: scales, biases: biases,
                    rhsIndices: indices, transpose: true, groupSize: 32, bits: bits)
            }
            return quantizedMM(
                x, weight, scales: scales, biases: biases,
                transpose: true, groupSize: 32, bits: bits)
        }
        let w = weight.asType(.float32).swappedAxes(-1, -2)
        if let indices { return gatherMM(x, w, rhsIndices: indices) }
        return matmul(x, w)
    }
}

struct Zonos3Component {
    let metadata: [String: Any]
    let params: Zonos3TensorStore
    init(_ directory: URL, architecture: String) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let metadata = object["metadata"] as? [String: Any],
            metadata["general.architecture"] as? String == architecture,
            let tensors = object["tensors"]
        else {
            throw Zonos3Error.invalidCheckpoint(
                "Expected \(architecture) in \(directory.lastPathComponent)")
        }
        self.metadata = metadata
        let specs = try JSONDecoder().decode(
            [String: Zonos3TensorSpecification].self,
            from: JSONSerialization.data(withJSONObject: tensors))
        params = try Zonos3TensorStore(
            specs: specs,
            weights: loadArrays(url: directory.appendingPathComponent("model.safetensors")))
    }
}
