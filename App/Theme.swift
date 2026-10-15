import SwiftUI

enum Theme {
    static let bg = Color(red: 0.025, green: 0.055, blue: 0.08)
    static let panel = Color(red: 0.055, green: 0.09, blue: 0.12)
    static let ink = Color(red: 0.94, green: 0.96, blue: 0.94)
    static let muted = Color(red: 0.59, green: 0.68, blue: 0.70)
    static let mint = Color(red: 0.47, green: 0.91, blue: 0.81)
    static let coral = Color(red: 1, green: 0.55, blue: 0.48)
    static let violet = Color(red: 0.68, green: 0.61, blue: 1)
    static let gold = Color(red: 0.96, green: 0.80, blue: 0.48)
}

struct SectionEyebrow: View {
    let text: String
    var body: some View {
        Text(text.uppercased()).font(.system(.caption2, design: .monospaced).weight(.semibold))
            .tracking(2).foregroundStyle(Theme.muted)
    }
}

struct InstrumentPanel<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        content.padding(18).frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.panel, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).stroke(.white.opacity(0.06), lineWidth: 1))
    }
}

struct ValueReadout: View {
    let title: String
    let value: String
    var color: Color = Theme.mint
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(Theme.muted)
            Text(value).font(.system(.title3, design: .monospaced).weight(.medium)).foregroundStyle(color)
        }.accessibilityElement(children: .combine)
    }
}

struct GridBackdrop: View {
    var body: some View {
        Canvas { context, size in
            var path = Path()
            for x in stride(from: 0.0, to: size.width, by: 28) {
                path.move(to: CGPoint(x: x, y: 0)); path.addLine(to: CGPoint(x: x, y: size.height))
            }
            for y in stride(from: 0.0, to: size.height, by: 28) {
                path.move(to: CGPoint(x: 0, y: y)); path.addLine(to: CGPoint(x: size.width, y: y))
            }
            context.stroke(path, with: .color(Theme.mint.opacity(0.025)), lineWidth: 0.5)
        }.allowsHitTesting(false).accessibilityHidden(true)
    }
}

struct TokenStrip: View {
    let tokens: [String]
    @Binding var selected: Int
    var tint: Color = Theme.mint
    var onSelect: ((Int) -> Void)? = nil
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(tokens.enumerated()), id: \.offset) { index, token in
                    Button {
                        selected = index
                        onSelect?(index)
                    } label: {
                        VStack(spacing: 3) {
                            Text(token.replacingOccurrences(of: "\n", with: "↵"))
                                .font(.system(.subheadline, design: .monospaced).weight(.medium)).lineLimit(1)
                            Text(String(format: "%02d", index)).font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(selected == index ? Theme.bg.opacity(0.8) : Theme.muted)
                        }.padding(.horizontal, 13).frame(minHeight: 48)
                            .foregroundStyle(selected == index ? Theme.bg : Theme.ink)
                            .background(selected == index ? tint : Theme.panel, in: RoundedRectangle(cornerRadius: 13))
                    }.buttonStyle(.plain)
                        .accessibilityLabel("Token \(index + 1), \(token)")
                        .accessibilityAddTraits(selected == index ? [.isSelected] : [])
                }
            }.padding(.vertical, 3)
        }
    }
}

struct HeatmapView: View {
    let matrix: [[Double]]
    let tokens: [String]
    @Binding var selectedRow: Int
    var tint: Color = Theme.mint
    var causal = false
    var onSelect: ((Int, Int, Double) -> Void)? = nil
    var body: some View {
        let rowCount = max(matrix.count, 1)
        let columnCount = max(matrix.map(\.count).max() ?? 0, 1)
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("QUERY ↓").foregroundStyle(Theme.coral)
                Spacer()
                Text("KEY →").foregroundStyle(Theme.violet)
            }.font(.system(.caption2, design: .monospaced).weight(.medium))
            GeometryReader { geometry in
                let cell = min(geometry.size.width / CGFloat(columnCount), geometry.size.height / CGFloat(rowCount))
                let gridWidth = cell * CGFloat(columnCount)
                let gridHeight = cell * CGFloat(rowCount)
                Canvas { context, _ in
                    for (r, row) in matrix.enumerated() {
                        for (c, value) in row.enumerated() {
                            let rect = CGRect(x: CGFloat(c) * cell + 1, y: CGFloat(r) * cell + 1, width: max(1, cell - 2), height: max(1, cell - 2))
                            let intensity = max(0, min(1, sqrt(max(0, value))))
                            let color = causal && c > r ? Color.white.opacity(0.035) : tint.opacity(0.08 + intensity * 0.92)
                            context.fill(Path(roundedRect: rect, cornerRadius: min(4, cell / 6)), with: .color(color))
                        }
                    }
                    if matrix.indices.contains(selectedRow) {
                        let rect = CGRect(x: 0, y: CGFloat(selectedRow) * cell, width: gridWidth, height: cell)
                        context.stroke(Path(roundedRect: rect, cornerRadius: 5), with: .color(Theme.coral), lineWidth: 2)
                    }
                }
                .frame(width: gridWidth, height: gridHeight)
                .contentShape(Rectangle())
                .gesture(SpatialTapGesture().onEnded { value in
                    let r = min(rowCount - 1, max(0, Int(value.location.y / cell)))
                    let c = min(columnCount - 1, max(0, Int(value.location.x / cell)))
                    if matrix.indices.contains(r), matrix[r].indices.contains(c) {
                        selectedRow = r
                        onSelect?(r, c, matrix[r][c])
                    }
                })
                .accessibilityElement(children: .ignore)
                .accessibilityIdentifier("attention-matrix")
                .accessibilityLabel("Attention matrix, \(matrix.count) query rows and \(matrix.map(\.count).max() ?? 0) key columns. Selected query \(selectedRow + 1)")
                .accessibilityValue(accessibleRow)
                .accessibilityAdjustableAction { direction in
                    switch direction {
                    case .increment: selectedRow = min(rowCount - 1, selectedRow + 1)
                    case .decrement: selectedRow = max(0, selectedRow - 1)
                    @unknown default: break
                    }
                    if matrix.indices.contains(selectedRow),
                       let strongest = matrix[selectedRow].enumerated().max(by: { $0.element < $1.element }) {
                        onSelect?(selectedRow, strongest.offset, strongest.element)
                    }
                }
                .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
            }.aspectRatio(max(1, CGFloat(columnCount) / CGFloat(rowCount)), contentMode: .fit)
            HStack {
                Text("0").font(.caption2)
                LinearGradient(colors: [tint.opacity(0.08), tint], startPoint: .leading, endPoint: .trailing)
                    .frame(width: 72, height: 4).clipShape(Capsule())
                Text("1").font(.caption2)
                Spacer()
                Text(causal ? "FUTURE TOKENS MASKED" : "EVERY TOKEN CAN CONNECT")
                    .font(.system(.caption2, design: .monospaced)).lineLimit(1).minimumScaleFactor(0.7)
            }.foregroundStyle(Theme.muted)
        }
    }
    private var accessibleRow: String {
        guard matrix.indices.contains(selectedRow) else { return "No attention values" }
        return matrix[selectedRow].enumerated().map { c, v in
            "\(tokens.indices.contains(c) ? tokens[c] : String(c)): \(Int(v * 100)) percent"
        }.joined(separator: ", ")
    }
}
