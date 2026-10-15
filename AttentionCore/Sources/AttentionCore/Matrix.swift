import Foundation

/// A small row-major matrix for inspectable, deterministic CPU experiments.
/// Dimension checks are programmer preconditions; user input is sanitized by TransformerEngine.
public struct Matrix: Equatable, Sendable {
    public let rows: Int
    public let columns: Int
    public var values: [Double]

    public init(rows: Int, columns: Int, values: [Double]) {
        precondition(rows > 0 && columns > 0 && values.count == rows * columns)
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    public init(_ rows: [[Double]]) {
        precondition(!rows.isEmpty && !(rows.first?.isEmpty ?? true))
        precondition(rows.allSatisfy { $0.count == rows[0].count })
        self.init(rows: rows.count, columns: rows[0].count, values: rows.flatMap { $0 })
    }

    public subscript(_ row: Int, _ column: Int) -> Double {
        get { values[row * columns + column] }
        set { values[row * columns + column] = newValue }
    }

    public func row(_ index: Int) -> [Double] {
        Array(values[(index * columns)..<((index + 1) * columns)])
    }

    public var transposed: Matrix {
        Matrix(rows: columns, columns: rows, values: (0..<columns).flatMap { c in
            (0..<rows).map { r in self[r, c] }
        })
    }

    public var maximumMagnitude: Double { values.filter(\.isFinite).map(abs).max() ?? 0 }

    public func multiplied(by other: Matrix) -> Matrix {
        precondition(columns == other.rows)
        var result = Matrix.zeros(rows: rows, columns: other.columns)
        for r in 0..<rows {
            for k in 0..<columns {
                for c in 0..<other.columns { result[r, c] += self[r, k] * other[k, c] }
            }
        }
        return result
    }

    public func adding(_ other: Matrix) -> Matrix {
        precondition(rows == other.rows && columns == other.columns)
        return Matrix(rows: rows, columns: columns, values: zip(values, other.values).map(+))
    }

    public func scaled(by factor: Double) -> Matrix {
        Matrix(rows: rows, columns: columns, values: values.map { $0 * factor })
    }

    public func addingRow(_ bias: [Double]) -> Matrix {
        precondition(bias.count == columns)
        var result = self
        for r in 0..<rows { for c in 0..<columns { result[r, c] += bias[c] } }
        return result
    }

    public static func zeros(rows: Int, columns: Int) -> Matrix {
        Matrix(rows: rows, columns: columns, values: Array(repeating: 0, count: rows * columns))
    }

    public static func concatenateColumns(_ matrices: [Matrix]) -> Matrix {
        precondition(!matrices.isEmpty && matrices.allSatisfy { $0.rows == matrices[0].rows })
        return Matrix(rows: matrices[0].rows, columns: matrices.reduce(0) { $0 + $1.columns },
                      values: (0..<matrices[0].rows).flatMap { r in matrices.flatMap { $0.row(r) } })
    }
}

public enum TransformerMath {
    /// Numerically stable row-wise softmax. Masked -infinity entries become exact zeros.
    /// An entirely masked row is represented by zeros, never NaNs or an invented distribution.
    public static func softmax(_ matrix: Matrix, temperature: Double = 1) -> Matrix {
        let temperature = temperature.isFinite ? max(0.05, temperature) : 1
        var result = Matrix.zeros(rows: matrix.rows, columns: matrix.columns)
        for r in 0..<matrix.rows {
            let values = matrix.row(r)
            let positiveInfinityCount = values.filter { $0 == .infinity }.count
            if positiveInfinityCount > 0 {
                for c in 0..<matrix.columns {
                    result[r, c] = values[c] == .infinity ? 1 / Double(positiveInfinityCount) : 0
                }
                continue
            }
            guard let maximum = values.filter(\.isFinite).max() else { continue }
            let exponents = values.map { $0.isFinite ? exp(($0 - maximum) / temperature) : 0 }
            let denominator = exponents.reduce(0, +)
            for c in 0..<matrix.columns { result[r, c] = exponents[c] / denominator }
        }
        return result
    }

    /// Original postnorm uses a separate LayerNorm after adding the residual.
    /// This demonstration uses gamma=1 and beta=0, across features within each token.
    public static func layerNormalize(_ matrix: Matrix, epsilon: Double = 1e-6) -> Matrix {
        var result = Matrix.zeros(rows: matrix.rows, columns: matrix.columns)
        for r in 0..<matrix.rows {
            let values = matrix.row(r)
            let mean = values.reduce(0, +) / Double(matrix.columns)
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(matrix.columns)
            let denominator = sqrt(variance + max(1e-12, epsilon))
            for c in 0..<matrix.columns { result[r, c] = (values[c] - mean) / denominator }
        }
        return result
    }

    public static func sinusoidalPositions(count: Int, dimension: Int, offset: Int = 0) -> Matrix {
        precondition(count > 0 && dimension > 0)
        return Matrix(rows: count, columns: dimension, values: (0..<count).flatMap { p in
            (0..<dimension).map { feature in
                let pair = feature / 2
                let angle = Double(p + offset) / pow(10_000, Double(2 * pair) / Double(dimension))
                return feature.isMultiple(of: 2) ? sin(angle) : cos(angle)
            }
        })
    }

    public static func entropy(_ probabilities: [Double]) -> Double {
        probabilities.reduce(0) { $0 - ($1 > 0 ? $1 * log($1) : 0) }
    }

    public static func causalMask(_ scores: Matrix) -> Matrix {
        var masked = scores
        for r in 0..<scores.rows { for c in 0..<scores.columns where c > r { masked[r, c] = -.infinity } }
        return masked
    }
}
