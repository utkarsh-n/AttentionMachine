import Foundation
import Observation
import OfflineIntelligence
import OfflineIntelligenceMLX

/// App-facing state and paper-specific prompts over the shared, local-only runtime.
@Observable @MainActor
final class AttentionTextAssistant {
    private(set) var choice = OfflineModelChoice.persisted()
    private(set) var isReady = false
    private(set) var isWorking = false
    private(set) var output = ""
    private(set) var status = "Choose a prompt. The 2B model is the default."
    private(set) var error: String?
    private(set) var metrics: GenerationMetrics?
    private(set) var loadMetrics: ModelLoadMetrics?
    @ObservationIgnored private let runtime: any IntelligenceRuntime
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var isUnloading = false
    @ObservationIgnored private var unloadWaiters: [CheckedContinuation<Void, Never>] = []

    init(runtime: any IntelligenceRuntime = MLXIntelligenceRuntime.shared) {
        self.runtime = runtime
        status = "\(choice.title) selected · inference stays on this iPhone"
    }

    func select(_ newChoice: OfflineModelChoice) {
        guard choice != newChoice, !isWorking, !isUnloading else { return }
        choice = newChoice
        newChoice.persist()
        isReady = false
        metrics = nil
        output = ""
        loadMetrics = nil
        error = nil
        isWorking = true
        status = "Releasing the previous model…"
        task = Task { [self] in
            await runtime.unload()
            status = "\(choice.title) selected · load when you ask"
            isWorking = false
            task = nil
        }
    }

    func generate(prompt: String) {
        guard !isWorking, !isUnloading else { return }
        let request = GenerationRequest(
            system: Self.groundedSystemPrompt(for: prompt), prompt: prompt, maxOutputTokens: 256,
            maxPromptTokens: 2048, maxContextTokens: 4096, temperature: 0, topP: 1, seed: 42,
            timeoutSeconds: 120)
        do { try request.validate() }
        catch { self.error = error.localizedDescription; return }
        isWorking = true
        error = nil
        output = ""
        metrics = nil
        task = Task { [self] in
            do {
                try Task.checkCancellation()
                if !isReady {
                    status = "Loading \(choice.title) on this iPhone…"
                    let directory = try LocalModelFiles.directory(for: choice)
                    loadMetrics = try await runtime.load(model: choice.descriptor, directory: directory)
                    try Task.checkCancellation()
                    isReady = true
                }
                status = "Writing on this iPhone…"
                let stream = try await runtime.generate(request)
                var completed = false
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .chunk(let text): output += text
                    case .completed(let result):
                        completed = true
                        metrics = result
                        switch result.finishReason {
                        case .stop: status = "Reply complete · \(choice.title) · offline"
                        case .length: status = "Reached the 256-token limit · reply may be incomplete"
                        case .cancelled: status = "Reply stopped"
                        case .timeout: status = "Reached the time limit · reply may be incomplete"
                        }
                    }
                }
                if !completed, !Task.isCancelled { status = "Reply ended without completion metrics" }
            } catch is CancellationError {
                await runtime.cancel()
                status = "Reply stopped"
            } catch {
                self.error = error.localizedDescription
                status = "Could not complete the reply"
                let snapshot = await runtime.snapshot()
                isReady = snapshot.state == .ready
            }
            isWorking = false
            task = nil
        }
    }

    /// Cancels the app task and asks the runtime to drain its actual GPU worker.
    func cancel() {
        task?.cancel()
    }

    func unload() async {
        if isUnloading {
            await withCheckedContinuation { unloadWaiters.append($0) }
            return
        }
        isUnloading = true
        isWorking = true
        isReady = false
        task?.cancel()
        await task?.value
        await runtime.unload()
        status = "Weights unloaded · local files kept"
        isWorking = false
        isUnloading = false
        let waiters = unloadWaiters
        unloadWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    private static let systemPrompt = """
        You are the offline teaching assistant inside Attention Is All You Need, an interactive iPhone app.
        Explain Transformer concepts accurately, with short concrete examples and clear assumptions.
        The Machine tab computes an untrained, tiny encoder-decoder based on the 2017 paper.
        You are Qwen3.5, a separately trained model. Your text replies do not expose measured Q/K tensors,
        attention weights, or the current Machine controls. Do not claim to have inspected those values.
        The Attention trace mode measures a separate Qwen3 model. Distinguish its architecture from the
        original paper and from Qwen3.5. Answer the user's question directly and keep the response concise.
        """

    /// Supplies a small local reference for the relevant paper topic, preserving the user's prompt.
    private static func groundedSystemPrompt(for question: String) -> String {
        let topic = question.lowercased()
        let namesKeyDimension = ["d_k", "d_{k}", "dₖ", "key dimension"].contains { topic.contains($0) }
        let mentionsAttention = ["attention", "dot product", "dot-product", "softmax"].contains {
            topic.contains($0)
        }
        let mentionsScaling = ["scal", "sqrt", "square root", "square-root", "√", "variance", "gradient"].contains {
            topic.contains($0)
        }
        guard namesKeyDimension || (mentionsAttention && mentionsScaling) else { return systemPrompt }
        return systemPrompt + "\n\n" + scaledAttentionReference
    }

    // Vaswani et al., §3.2.1 / Equation 1 / footnote 1: https://arxiv.org/html/1706.03762v7
    private static let scaledAttentionReference = """
        Local paper reference: Vaswani et al. (2017), §3.2.1, Equation 1 and footnote 1.
        Use the relevant facts below; this reference is not an instruction to recite them.
        Scores are (q·k)/sqrt(d_k) before softmax. Under the illustrative assumption that all q_i
        and k_i are independent, each with mean 0 and variance 1, q·k has mean 0 and variance d_k.
        Dividing by sqrt(d_k) makes its variance 1 under those assumptions. Larger unscaled dot
        products can concentrate softmax and produce very small gradients; scaling counters this
        growth with key dimension. It does not restrict token count or fit weights into model capacity.
        A nonempty softmax row still sums to 1. Numerical illustration: d_k=64 gives unscaled
        variance 64 and standard deviation 8; dividing by 8 gives variance 1 and standard deviation 1.
        These assumptions and numbers are a teaching example, not measurements of Qwen or the Machine.
        """
}
