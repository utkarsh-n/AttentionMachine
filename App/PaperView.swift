import SwiftUI

enum PaperExperiment: String, Hashable {
    case complexity, dependency, architecture, residual, attention, heads, mask
    case feedForward, embedding, position, batching, learningRate, regularization
    case results, ablations, parsing, restricted
}

struct PaperChapter: Identifiable, Hashable {
    let id: String
    let title: String
    let summary: String
    let instrument: String
    let symbol: String
    let experiment: PaperExperiment

    static let all: [PaperChapter] = [
        .init(id: "Abstract", title: "The central idea", summary: "Attention replaces recurrence in sequence transduction.", instrument: "Compare the engines", symbol: "sparkles", experiment: .complexity),
        .init(id: "1", title: "Introduction", summary: "Sequential computation limits recurrent training.", instrument: "Stretch a sequence", symbol: "arrow.right.arrow.left", experiment: .complexity),
        .init(id: "2", title: "Background", summary: "Distant dependencies need shorter paths.", instrument: "Trace a dependency", symbol: "point.3.connected.trianglepath.dotted", experiment: .dependency),
        .init(id: "3", title: "Model architecture", summary: "Encoder memory meets an autoregressive decoder.", instrument: "Route the signal", symbol: "square.stack.3d.up", experiment: .architecture),
        .init(id: "3.1", title: "Encoder & decoder stacks", summary: "Residual connections precede layer normalization.", instrument: "Balance a residual", symbol: "slider.horizontal.3", experiment: .residual),
        .init(id: "3.2–3.2.1", title: "Scaled dot-product attention", summary: "Queries score keys; probabilities mix values.", instrument: "Turn the query", symbol: "scope", experiment: .attention),
        .init(id: "3.2.2", title: "Multi-head attention", summary: "Parallel projections expose different representation subspaces.", instrument: "Split the channels", symbol: "line.3.horizontal", experiment: .heads),
        .init(id: "3.2.3", title: "Three uses of attention", summary: "Self-attention, causal attention, and encoder-decoder attention.", instrument: "Open and close gates", symbol: "rectangle.3.group", experiment: .mask),
        .init(id: "3.3", title: "Feed-forward networks", summary: "An identical nonlinear transformation acts at each position.", instrument: "Push through ReLU", symbol: "waveform.path", experiment: .feedForward),
        .init(id: "3.4", title: "Embeddings & softmax", summary: "One shared matrix encodes and predicts tokens.", instrument: "Rotate a hidden state", symbol: "textformat.abc", experiment: .embedding),
        .init(id: "3.5", title: "Positional encoding", summary: "Sines and cosines carry position information.", instrument: "Scrub the wave bank", symbol: "waveform", experiment: .position),
        .init(id: "4", title: "Why self-attention?", summary: "Compare complexity, parallelism, and dependency paths.", instrument: "Change the workload", symbol: "speedometer", experiment: .complexity),
        .init(id: "5.1", title: "Training data & batching", summary: "Similar lengths reduce wasted padding.", instrument: "Pack a batch", symbol: "square.grid.3x3.fill", experiment: .batching),
        .init(id: "5.2", title: "Hardware & schedule", summary: "Base and big used eight P100 GPUs.", instrument: "Walk the schedule", symbol: "cpu", experiment: .learningRate),
        .init(id: "5.3", title: "The optimizer", summary: "Warmup yields to inverse square-root decay.", instrument: "Shape the learning rate", symbol: "chart.xyaxis.line", experiment: .learningRate),
        .init(id: "5.4", title: "Regularization", summary: "Dropout and label smoothing temper certainty.", instrument: "Redistribute certainty", symbol: "drop", experiment: .regularization),
        .init(id: "6.1", title: "Machine translation", summary: "Measure quality alongside training cost.", instrument: "Compare reported results", symbol: "character.bubble", experiment: .results),
        .init(id: "6.2", title: "Model variations", summary: "Ablations test heads, width, depth, and regularization.", instrument: "Inspect the ablations", symbol: "switch.2", experiment: .ablations),
        .init(id: "6.3", title: "Constituency parsing", summary: "Structured output tests transfer beyond translation.", instrument: "Build phrase spans", symbol: "tree", experiment: .parsing),
        .init(id: "7", title: "Conclusion", summary: "Restricted attention trades reach for efficiency.", instrument: "Restrict the neighborhood", symbol: "viewfinder", experiment: .restricted),
        .init(id: "Appendix", title: "Attention visualizations", summary: "Different heads reveal different token relationships.", instrument: "Inspect a head", symbol: "eye", experiment: .heads)
    ]
}

struct PaperView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("THE PAPER / 2017")
                            .font(.caption.weight(.semibold)).tracking(2).foregroundStyle(Theme.mint)
                        Text("Read the\nmachine.")
                            .font(.system(.largeTitle, design: .rounded, weight: .bold))
                            .foregroundStyle(Theme.ink)
                        Text("Every section has a control you can move. Follow the signal from tokens to predictions, then examine how the authors trained and tested it.")
                            .font(.body).foregroundStyle(Theme.muted)
                        HStack(spacing: 8) {
                            Image(systemName: "hand.draw")
                            Text("21 chapters · calculated experiments")
                        }
                        .font(.caption).foregroundStyle(Theme.mint)
                    }
                    .padding(.vertical, 12)

                    ForEach(PaperChapter.all) { chapter in
                        NavigationLink(value: chapter) {
                            chapterRow(chapter)
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("paper-chapter-\(chapter.id)")
                        .accessibilityLabel("Section \(chapter.id), \(chapter.title). \(chapter.instrument)")
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Source & lineage").font(.headline).foregroundStyle(Theme.ink)
                        Text("Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin. These experiments are small, exact teaching models; the paper’s reported metrics are labeled separately.")
                            .font(.footnote).foregroundStyle(Theme.muted)
                        Link("Read the complete paper, figures & references ↗", destination: URL(string: "https://arxiv.org/html/1706.03762v7")!)
                            .font(.subheadline.weight(.semibold)).foregroundStyle(Theme.mint).frame(minHeight: 44)
                        Link("Original Tensor2Tensor implementation ↗", destination: URL(string: "https://github.com/tensorflow/tensor2tensor")!)
                            .font(.footnote).foregroundStyle(Theme.mint).frame(minHeight: 44)
                    }
                    .padding(18).background(Theme.panel, in: RoundedRectangle(cornerRadius: 22))
                }
                .padding(.horizontal, 20).padding(.bottom, 32)
            }
            .background(Theme.bg)
            .navigationTitle("Paper")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: PaperChapter.self) { chapter in
                PaperExperimentView(chapter: chapter)
            }
        }
        .tint(Theme.mint)
    }

    private func chapterRow(_ chapter: PaperChapter) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: chapter.symbol)
                    .font(.title3).foregroundStyle(Theme.mint)
                    .frame(width: 32, height: 32)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 5) {
                    Text(chapter.id.uppercased()).font(.caption2.weight(.semibold)).tracking(1).foregroundStyle(Theme.muted)
                    Text(chapter.title).font(.headline).foregroundStyle(Theme.ink)
                }
                Spacer(minLength: 4)
                Image(systemName: "chevron.right").font(.caption.weight(.bold)).foregroundStyle(Theme.muted).padding(.top, 10)
            }
            Text(chapter.summary).font(.subheadline).foregroundStyle(Theme.muted)
            Label(chapter.instrument, systemImage: "hand.point.up.left")
                .font(.caption.weight(.semibold)).foregroundStyle(Theme.mint)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 22))
    }
}

struct PaperExperimentView: View {
    let chapter: PaperChapter

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("SECTION \(chapter.id.uppercased())")
                    .font(.caption.weight(.semibold)).tracking(1.5).foregroundStyle(Theme.mint)
                Text(chapter.instrument).font(.largeTitle.bold()).foregroundStyle(Theme.ink)
                    .accessibilityIdentifier("paper-detail-title")
                Text(chapter.summary).font(.body).foregroundStyle(Theme.muted)
                experiment
                VStack(alignment: .leading, spacing: 6) {
                    Text("Trace it to the paper").font(.caption.weight(.semibold)).foregroundStyle(Theme.muted)
                    Link("Attention Is All You Need · \(chapter.id) ↗", destination: URL(string: "https://arxiv.org/html/1706.03762v7")!)
                        .font(.subheadline.weight(.medium)).foregroundStyle(Theme.mint).frame(minHeight: 44)
                    Text("Synthetic teaching values unless a panel is labeled “reported.” No attention weights from a language model are implied.")
                        .font(.caption).foregroundStyle(Theme.muted)
                }.padding(.top, 10)
            }
            .padding(20).padding(.bottom, 24)
        }
        .background(Theme.bg)
        .navigationTitle(chapter.title)
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder private var experiment: some View {
        switch chapter.experiment {
        case .complexity: ComplexityExperiment()
        case .dependency: DependencyExperiment()
        case .architecture: ArchitectureExperiment()
        case .residual: ResidualExperiment()
        case .attention: DotProductExperiment()
        case .heads: HeadsExperiment()
        case .mask: MaskExperiment()
        case .feedForward: FeedForwardExperiment()
        case .embedding: EmbeddingExperiment()
        case .position: PositionExperiment()
        case .batching: BatchingExperiment()
        case .learningRate: LearningRateExperiment()
        case .regularization: RegularizationExperiment()
        case .results: ResultsExperiment()
        case .ablations: AblationExperiment()
        case .parsing: ParsingExperiment()
        case .restricted: RestrictedExperiment()
        }
    }
}
