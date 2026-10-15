import Foundation
import Observation
import CryptoKit
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Tokenizers
import OfflineIntelligenceMLX
#if DEBUG
import UIKit
#endif

struct LocalToken: Codable, Sendable, Identifiable {
    let index: Int
    let id: Int
    let text: String
    let isSpecial: Bool
}

struct LocalNextToken: Codable, Sendable, Identifiable {
    let id: Int
    let text: String
    let probability: Double
}

struct LocalModelMetadata: Codable, Sendable {
    let id: String
    let revision: String
    let architecture: String
    let quantization: String
    let layers: Int
    let heads: Int
    let kvHeads: Int
    let headDimension: Int
    let license: String

    static let qwen = LocalModelMetadata(
        id: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        revision: "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
        architecture: "qwen3", quantization: "4-bit affine, group size 64",
        layers: 36, heads: 32, kvHeads: 8, headDimension: 128, license: "Apache-2.0")
}

struct LocalModelMetrics: Codable, Sendable {
    let elapsedSeconds: Double
    let rowSumMaxError: Double
    let maskedFutureMax: Double
}

struct LocalModelProvenance: Codable, Sendable {
    let method: String
    let source: String
    let runtime: String
    let weightSHA256: String
}

struct LocalModelTrace: Codable, Sendable {
    let schemaVersion: Int
    let mode: String
    let capturedAt: String
    let prompt: String
    let model: LocalModelMetadata
    let tokens: [LocalToken]
    let layer: Int
    let head: Int
    let attention: [[Double]]
    let nextTokens: [LocalNextToken]
    let generation: String
    let generatedTokenIDs: [Int]
    let metrics: LocalModelMetrics
    let provenance: LocalModelProvenance

    static func recorded() -> LocalModelTrace? {
        guard let url = Bundle.main.url(forResource: "RecordedTrace", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }
}

private enum LocalModelError: LocalizedError {
    case unsupportedSimulator, noModel, promptTooLong(Int), invalidSelection, invalidDownload, checksum
    var errorDescription: String? {
        switch self {
        case .unsupportedSimulator: "Live MLX inference needs a physical iPhone. The recorded exhibit is available here."
        case .noModel: "Download and load Qwen3 before running a live experiment."
        case .promptTooLong(let count): "This prompt has \(count) tokens including its chat template. Shorten it to 96 or fewer."
        case .invalidSelection: "Choose a layer from 1–36 and a head from 1–32."
        case .invalidDownload: "The model download returned an invalid file. Please retry."
        case .checksum: "A model file failed its SHA-256 check. Download it again."
        }
    }
}

/// All prompt processing, attention inspection and generation happen inside the app.
/// Network access is used only for the initial immutable public weight download.
@Observable @MainActor
final class LocalModelEngine {
    private(set) var isReady = false
    private(set) var isWorking = false
    private(set) var progress = 0.0
    private(set) var status = "Qwen3 · 4 billion parameters · 2.28 GB download"
    private(set) var error: String?
    private(set) var trace: LocalModelTrace?
    private(set) var hasDownloadedModel = false
    let layerCount = 36
    let headCount = 32
    let modelName = "Qwen3 4B Instruct"

    @ObservationIgnored private let worker = OnDeviceModelWorker()
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let directory: URL
    @ObservationIgnored private var isUnloading = false
    @ObservationIgnored private var unloadWaiters: [CheckedContinuation<Void, Never>] = []

    init() {
        directory = Self.modelDirectory
        hasDownloadedModel = Self.bundledModelDirectory != nil || FileManager.default.fileExists(atPath: directory.appendingPathComponent("verified.json").path)
        if hasDownloadedModel { status = "Qwen3 stored on this iPhone · load to begin" }
    }

    static var modelDirectory: URL {
        if let bundled = bundledModelDirectory { return bundled }
        return URL.applicationSupportDirectory.appendingPathComponent("AttentionMachine/Models/Qwen3-4B-Instruct-2507-4bit", isDirectory: true)
    }

    private static var bundledModelDirectory: URL? {
        let name = "Qwen3-4B-Instruct-2507-4bit"
        for folder in [Bundle.main.url(forResource: name, withExtension: nil),
                       Bundle.main.url(forResource: name, withExtension: nil, subdirectory: "Model"),
                       Bundle.main.url(forResource: "Model", withExtension: nil)] {
            if let folder, ModelFiles.files.allSatisfy({ file in
                let url = folder.appendingPathComponent(file.name)
                return (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) == file.bytes
            }) { return folder }
        }
        return nil
    }

    func download() async {
        guard !isWorking, !isUnloading else { return }
        if Self.bundledModelDirectory != nil { await load(); return }
        #if targetEnvironment(simulator)
        error = LocalModelError.unsupportedSimulator.localizedDescription
        return
        #else
        isWorking = true
        error = nil
        progress = 0
        status = "Downloading Qwen3 to this iPhone…"
        let directory = directory
        task = Task { [self] in
            do {
                try await ModelFiles.download(to: directory) { [weak self] fraction in
                    Task { @MainActor in self?.progress = fraction }
                }
                try Task.checkCancellation()
                hasDownloadedModel = true
                status = "Loading quantized weights on this iPhone…"
                try await worker.load(directory: directory)
                try Task.checkCancellation()
                isReady = true
                progress = 1
                status = "Qwen3 ready · inference stays on this iPhone"
            } catch is CancellationError {
                await worker.unload()
                isReady = false
                status = "Cancelled · completed download files are kept"
            } catch {
                self.error = error.localizedDescription
                status = "Model setup needs attention"
            }
            isWorking = false
            task = nil
        }
        await task?.value
        #endif
    }

    func load() async {
        guard !isWorking, !isUnloading, hasDownloadedModel else { return }
        #if targetEnvironment(simulator)
        error = LocalModelError.unsupportedSimulator.localizedDescription
        return
        #else
        isWorking = true
        error = nil
        status = "Loading Qwen3 on this iPhone…"
        task = Task { [self] in
            do {
                try await worker.load(directory: directory)
                try Task.checkCancellation()
                isReady = true
                status = "Qwen3 ready · inference stays on this iPhone"
            } catch is CancellationError {
                await worker.unload()
                isReady = false
                status = "Model loading cancelled"
            } catch {
                self.error = error.localizedDescription
                status = "Could not load model"
            }
            isWorking = false
            task = nil
        }
        await task?.value
        #endif
    }

    func inspect(prompt: String, layer: Int, head: Int, maxNewTokens: Int = 32) async {
        guard !isWorking, !isUnloading else { return }
        guard isReady else { error = LocalModelError.noModel.localizedDescription; return }
        isWorking = true
        error = nil
        status = "Tracing actual Q, K and attention on this iPhone…"
        task = Task { [self] in
            do {
                trace = try await worker.inspect(prompt: prompt, layer: layer, head: head, maxNewTokens: min(64, max(1, maxNewTokens)))
                status = "Live trace · \(trace?.tokens.count ?? 0) input tokens · on this iPhone"
            } catch is CancellationError {
                status = "Experiment cancelled"
            } catch {
                self.error = error.localizedDescription
                status = "Experiment needs attention"
            }
            isWorking = false
            task = nil
        }
        await task?.value
    }

    func cancel() { task?.cancel() }

    #if DEBUG
    /// Opt-in QA only. Reserve this engine while its worker drains Metal and
    /// reads allocator counters; an overlapping scene unload keeps ownership.
    func stabilityMemorySnapshot() async -> Memory.Snapshot? {
        guard !isWorking, !isUnloading, !Task.isCancelled,
              UIApplication.shared.applicationState == .active else { return nil }
        isWorking = true
        defer { if !isUnloading { isWorking = false } }
        let snapshot = await worker.stabilityMemorySnapshot()
        guard !isUnloading, !Task.isCancelled,
              UIApplication.shared.applicationState == .active else { return nil }
        return snapshot
    }
    #endif

    func unload() async {
        if isUnloading {
            await withCheckedContinuation { unloadWaiters.append($0) }
            return
        }
        isUnloading = true
        isWorking = true
        isReady = false
        defer {
            isWorking = false; isUnloading = false
            let waiters = unloadWaiters; unloadWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
        }
        task?.cancel()
        await task?.value
        isWorking = true
        await worker.unload()
        status = "Weights unloaded · download kept on this iPhone"
    }
}

private actor OnDeviceModelWorker {
    private let residencyLease = MLXExternalModelLease()
    private var ownsResidency = false
    private var model: InstrumentedQwen3Model?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private let probe = AttentionCaptureBox()
    private var specialTokenIDs = Set<Int>()

    func load(directory: URL) async throws {
        try await MLXForegroundExecution.prepare()
        if model != nil { return }
        try checkGPUAdmission()
        try residencyLease.acquire()
        ownsResidency = true
        do {
            Memory.cacheLimit = 16 * 1024 * 1024
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(InstrumentedQwen3Configuration.self, from: data)
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
            // Scope disposable placeholder RNG locally; strict checkpoint loading
            // replaces every parameter before any inference uses this model.
            let candidate = withRandomState(MLXRandom.RandomState(seed: 2017)) {
                InstrumentedQwen3Model(configuration, probe: probe)
            }
            try checkGPUAdmission()
            try await loadWeights(modelDirectory: directory, model: candidate, perLayerQuantization: base.perLayerQuantization)
            try checkGPUAdmission()
            let loadedTokenizer = try await AutoTokenizer.from(modelFolder: directory)
            try checkGPUAdmission()
            let tokenizerData = try Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
            let metadata = try JSONDecoder().decode(TokenizerSpecialMetadata.self, from: tokenizerData)
            specialTokenIDs = Set(metadata.addedTokens.compactMap { key, token in
                token.special == true ? Int(key) : nil
            })
            model = candidate
            tokenizer = loadedTokenizer
        } catch {
            unload()
            throw error
        }
    }

    func unload() {
        guard ownsResidency else { return }
        // Root mode switching awaits the inspection task before this barrier; no new GPU work may overlap.
        MLXForegroundExecution.synchronizeIfAllowed()
        model = nil
        tokenizer = nil
        probe.matrix = nil
        specialTokenIDs.removeAll()
        Memory.clearCache()
        residencyLease.release()
        ownsResidency = false
    }

    #if DEBUG
    func stabilityMemorySnapshot() async -> Memory.Snapshot? {
        // Recheck after the actor hop. Stream synchronization can submit a
        // command buffer, so diagnostic barriers belong to foreground QA.
        guard await MainActor.run(body: { UIApplication.shared.applicationState == .active }),
              !Task.isCancelled else { return nil }
        MLXForegroundExecution.synchronizeIfAllowed()
        // Completion handlers can recycle buffers after an output event fires.
        // Clear these only after draining an unloaded model's stream.
        if model == nil { Memory.clearCache() }
        return Memory.snapshot()
    }
    #endif

    private func checkGPUAdmission() throws {
        try Task.checkCancellation()
        guard MLXForegroundExecution.isAllowed else { throw CancellationError() }
    }

    func inspect(prompt: String, layer: Int, head: Int, maxNewTokens: Int) throws -> LocalModelTrace {
        guard let model, let tokenizer else { throw LocalModelError.noModel }
        guard (0..<36).contains(layer), (0..<32).contains(head) else { throw LocalModelError.invalidSelection }
        let start = Date()
        let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        let tokenIDs = tokenizer.encode(text: formatted, addSpecialTokens: false)
        guard tokenIDs.count <= 96 else { throw LocalModelError.promptTooLong(tokenIDs.count) }
        try checkGPUAdmission()
        probe.layer = layer
        probe.head = head
        probe.enabled = true
        probe.matrix = nil
        defer { probe.enabled = false; probe.matrix = nil; Memory.clearCache() }
        let logits = model(MLXArray(tokenIDs).reshaped(1, -1), cache: nil)[0, -1]
        try checkGPUAdmission()
        eval(logits)
        guard let captured = probe.matrix else { throw LocalModelError.noModel }
        try checkGPUAdmission()
        let flat = captured.asType(.float32).asArray(Float.self).map(Double.init)
        let count = tokenIDs.count
        let attention = (0..<count).map { Array(flat[($0 * count)..<(($0 + 1) * count)]) }
        try checkGPUAdmission()
        let distribution = softmax(logits.asType(.float32)).asArray(Float.self)
        let topIDs = distribution.indices.sorted { distribution[$0] > distribution[$1] }.prefix(8)
        let next = topIDs.map { LocalNextToken(id: $0, text: tokenizer.decode(tokens: [$0]), probability: Double(distribution[$0])) }
        probe.enabled = false
        try checkGPUAdmission()
        // A fresh KV cache keeps generation bounded and avoids recomputing the entire prefix each step.
        let cache = try model.newCache(parameters: nil)
        var generationLogits = model(MLXArray(tokenIDs).reshaped(1, -1), cache: cache)[0, -1]
        var generated = [Int]()
        for _ in 0..<maxNewTokens {
            try checkGPUAdmission()
            let nextID = generationLogits.argMax().item(Int.self)
            if nextID == 151645 || nextID == 151643 { break }
            generated.append(nextID)
            if generated.count < maxNewTokens {
                generationLogits = model(MLXArray([nextID]).reshaped(1, 1), cache: cache)[0, -1]
            }
        }
        let rowsError = attention.map { abs($0.reduce(0, +) - 1) }.max() ?? 0
        let futureMax = attention.enumerated().flatMap { i, row in Array(row.dropFirst(i + 1)) }.max() ?? 0
        return LocalModelTrace(schemaVersion: 1, mode: "live", capturedAt: ISO8601DateFormatter().string(from: Date()), prompt: prompt,
            model: .qwen, tokens: tokenIDs.enumerated().map { index, id in
                LocalToken(index: index, id: id, text: tokenizer.decode(tokens: [id]), isSpecial: specialTokenIDs.contains(id))
            }, layer: layer, head: head, attention: attention, nextTokens: next,
            generation: tokenizer.decode(tokens: generated, skipSpecialTokens: true), generatedTokenIDs: generated,
            metrics: .init(elapsedSeconds: Date().timeIntervalSince(start), rowSumMaxError: rowsError, maskedFutureMax: futureMax),
            provenance: .init(method: "Float32 causal softmax from actual Qwen3 Q/K tensors after RMSNorm and RoPE; fused attention path unchanged",
                source: "https://huggingface.co/\(LocalModelMetadata.qwen.id)", runtime: "MLX Swift · on this iPhone", weightSHA256: ModelFiles.weightSHA))
    }
}

private enum ModelFiles {
    static let weightSHA = "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f"
    static let files: [(name: String, bytes: Int64, sha: String?)] = [
        ("config.json", 938, "574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849"),
        ("tokenizer_config.json", 5440, "4397cc477eb6d79715ccd2000accd6b3531928f30029665832fa1b255f24d2b9"),
        ("tokenizer.json", 11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        ("model.safetensors", 2_263_022_417, weightSHA)
    ]

    static func download(to directory: URL, progress: @Sendable @escaping (Double) -> Void) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var resource = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try resource.setResourceValues(values)
        let total = files.reduce(Int64(0)) { $0 + $1.bytes }
        var completed: Int64 = 0
        for file in files {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(file.name)
            let size = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init)
            if size == file.bytes {
                if let expected = file.sha, try checksum(destination) != expected {
                    try FileManager.default.removeItem(at: destination)
                } else {
                    completed += file.bytes
                    progress(Double(completed) / Double(total))
                    continue
                }
            }
            let url = URL(string: "https://huggingface.co/\(LocalModelMetadata.qwen.id)/resolve/\(LocalModelMetadata.qwen.revision)/\(file.name)")!
            let prior = completed
            let downloader = SingleFileDownload()
            let temporary = try await downloader.download(url: url) { bytes in
                progress(min(1, Double(prior + bytes) / Double(total)))
            }
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard (try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) == file.bytes else {
                throw LocalModelError.invalidDownload
            }
            if let expected = file.sha, try checksum(temporary) != expected { throw LocalModelError.checksum }
            if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
            try FileManager.default.moveItem(at: temporary, to: destination)
            completed += file.bytes
            progress(Double(completed) / Double(total))
        }
        let manifest = try JSONEncoder().encode(LocalModelMetadata.qwen)
        try manifest.write(to: directory.appendingPathComponent("verified.json"), options: .atomic)
    }

    private static func checksum(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty {
            try Task.checkCancellation()
            digest.update(data: data)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private struct TokenizerSpecialMetadata: Decodable {
    struct Token: Decodable { let special: Bool? }
    let addedTokens: [String: Token]
    enum CodingKeys: String, CodingKey { case addedTokens = "added_tokens_decoder" }
}

private final class SingleFileDownload: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<URL, any Error>?
    private var task: URLSessionDownloadTask?
    private var session: URLSession?
    private var report: (@Sendable (Int64) -> Void)?
    private var cancelled = false

    func download(url: URL, progress: @Sendable @escaping (Int64) -> Void) async throws -> URL {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                if cancelled { lock.unlock(); continuation.resume(throwing: CancellationError()); return }
                self.continuation = continuation
                report = progress
                let configuration = URLSessionConfiguration.default
                configuration.timeoutIntervalForResource = 3600
                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
                self.session = session
                let task = session.downloadTask(with: url)
                self.task = task
                lock.unlock()
                task.resume()
            }
        } onCancel: { [self] in
            lock.lock(); cancelled = true; let current = task; lock.unlock()
            current?.cancel()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        lock.lock(); let callback = report; lock.unlock()
        callback?(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            guard let response = downloadTask.response as? HTTPURLResponse, response.statusCode == 200 else {
                throw LocalModelError.invalidDownload
            }
            let persistent = URL.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.moveItem(at: location, to: persistent)
            finish(.success(persistent))
        } catch { finish(.failure(error)) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if let error {
            finish(.failure((error as NSError).code == NSURLErrorCancelled ? CancellationError() : error))
        }
    }

    private func finish(_ result: Result<URL, any Error>) {
        lock.lock(); let waiting = continuation; continuation = nil; let session = session; self.session = nil; task = nil; lock.unlock()
        waiting?.resume(with: result)
        session?.finishTasksAndInvalidate()
    }
}

private final class AttentionCaptureBox {
    var enabled = false
    var layer = 0
    var head = 0
    var matrix: MLXArray?
}

/* Adapted from ml-explore/mlx-swift-lm Qwen3.swift at tag 3.32.3.
MIT License

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
private class InstrumentedQwen3Attention: Module {
    let args: InstrumentedQwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    let probe: AttentionCaptureBox
    let layerIndex: Int

    init(_ args: InstrumentedQwen3Configuration, probe: AttentionCaptureBox, layerIndex: Int) {
        self.probe = probe
        self.layerIndex = layerIndex
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: headDim,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE positioning
        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // Capture selected-head probabilities from the actual Q/K tensors.
        // Inference continues through the untouched fused attention path below.
        if probe.enabled && probe.layer == layerIndex && cache == nil {
            let kvHead = probe.head / (args.attentionHeads / args.kvHeads)
            let q = queries[0, probe.head].asType(.float32)
            let k = keys[0, kvHead].asType(.float32)
            let scores = matmul(q, k.transposed()) * scale
            let positions = arange(L)
            let causal = positions.expandedDimensions(axis: 1) .>= positions.expandedDimensions(axis: 0)
            probe.matrix = softmax(which(causal, scores, -Float.infinity), axis: -1)
        }

        // Use the automatic attention router that handles both quantized and regular caches
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class InstrumentedQwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class InstrumentedQwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: InstrumentedQwen3Attention
    let mlp: InstrumentedQwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: InstrumentedQwen3Configuration, probe: AttentionCaptureBox, layerIndex: Int) {
        _attention.wrappedValue = InstrumentedQwen3Attention(args, probe: probe, layerIndex: layerIndex)
        self.mlp = InstrumentedQwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

private class InstrumentedQwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [InstrumentedQwen3TransformerBlock]
    let norm: RMSNorm

    init(_ args: InstrumentedQwen3Configuration, probe: AttentionCaptureBox) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { index in
                InstrumentedQwen3TransformerBlock(args, probe: probe, layerIndex: index)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// This model has one owning OnDeviceModelWorker actor. During initial loading,
// MLX's @concurrent loadWeights materializes its tensors before the worker stores
// the candidate or allows inference. Afterwards every forward pass and probe
// mutation is actor-confined. This narrow Sendable declaration permits that
// exclusive loading handoff; the model is never published to the UI or shared
// between inference tasks.
private final class InstrumentedQwen3Model: Module, LLMModel, KVCacheDimensionProvider, @unchecked Sendable {
    let vocabularySize: Int
    let kvHeads: [Int]
    var loraLayers: [Module] { model.layers }

    let model: InstrumentedQwen3ModelInner
    let configuration: InstrumentedQwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: InstrumentedQwen3Configuration, probe: AttentionCaptureBox) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = InstrumentedQwen3ModelInner(args, probe: probe)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights = weights.filter { !$0.key.hasPrefix("lm_head.") }
        }

        return weights
    }
}

private struct InstrumentedQwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    init(from decoder: any Swift.Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<InstrumentedQwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: InstrumentedQwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: InstrumentedQwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: InstrumentedQwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: InstrumentedQwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: InstrumentedQwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

