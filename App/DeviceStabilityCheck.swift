import Foundation
import Darwin
import MLX
import Tokenizers
import AttentionCore
import UIKit

/// Development-only bounded QA. The caller opts in with --verify-stability.
/// Report data contains fixture identifiers, counts, metrics and memory only.
@MainActor
enum DeviceStabilityCheck {
    private static var hasRun = false
    private static var longHasRun = false
    private static let publicPrompt = "The animal did not cross the street because it was tired. What was tired?"
    private static let deadlineSeconds = 110.0

    /// Separate opt-in development mode: --verify-long-stability. Twenty minutes
    /// of foreground workload, with a twenty-one-minute cooperative deadline.
    static func runLong(engine: LocalModelEngine) async {
        guard !longHasRun else { return }
        longHasRun = true
        let previousIdleTimerSetting = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = previousIdleTimerSetting }
        let context = LongRunContext(engine: engine)
        context.save()
        print("ATTENTION_LONG_STABILITY started targetSeconds=1200 deadlineSeconds=1260")

        let lifecycleObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willResignActiveNotification, object: nil, queue: .main
        ) { _ in
            Task { @MainActor in
                context.stop(status: "interrupted", code: "foreground-interrupted")
            }
        }
        defer { NotificationCenter.default.removeObserver(lifecycleObserver) }

        let pressureSource = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .main)
        pressureSource.setEventHandler {
            let event = pressureSource.data
            Task { @MainActor in
                let critical = event.contains(.critical)
                let level = critical ? "critical" : (event.contains(.warning) ? "warning" : "normal")
                context.report.pressureEvents.append(LongPressureEvent(capturedAt: stamp(),
                    elapsedSeconds: context.elapsed, level: level))
                if critical { context.stop(status: "aborted", code: "critical-memory-pressure") }
                else { context.save() }
            }
        }
        pressureSource.resume()
        defer { pressureSource.setEventHandler {}; pressureSource.cancel() }

        let watchdog = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(1_260))
                context.stop(status: "timed_out", code: "cooperative-deadline-reached")
            } catch { /* Normal completion cancels this watchdog. */ }
        }
        let heartbeat = Task { @MainActor in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(2)) } catch { break }
                // MLX's allocator getters are not synchronized. Heartbeats read
                // Mach and thermal state only; foreground milestones drain the worker stream.
                context.sample(phase: "heartbeat")
                context.save()
                context.checkEnvironment()
            }
        }
        defer { watchdog.cancel(); heartbeat.cancel() }

        do {
            #if targetEnvironment(simulator)
            throw LongRunFailure(code: "physical-device-required")
            #else
            // SwiftUI may create the root before application activation completes.
            let activationDeadline = ProcessInfo.processInfo.systemUptime + 5
            while UIApplication.shared.applicationState != .active {
                try context.checkpoint()
                guard ProcessInfo.processInfo.systemUptime < activationDeadline else {
                    throw LongRunFailure(code: "foreground-not-active-at-start")
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            context.foregroundEstablished = true
            await engine.unload()
            try context.checkpoint()
            try await context.sampleMilestone(phase: "baseline")
            guard let baseline = context.report.samples.last?.memory.mlxActiveBytes else {
                throw LongRunFailure(code: "baseline-allocator-measurement-unavailable")
            }
            context.report.baselineMLXActiveBytes = baseline
            context.save()
            let fixtures = try await buildLongFixtures()
            guard fixtures.defaultCount == 24, (90...96).contains(fixtures.nearLimit.count),
                  (1...96).contains(fixtures.counting.count) else {
                throw LongRunFailure(code: "public-fixture-token-count-mismatch")
            }
            try context.checkpoint()
            context.report.phase = "cycling"
            while context.elapsed < context.report.targetDurationSeconds {
                try context.checkpoint()
                try await longCycle(engine: engine, fixtures: fixtures, context: context)
                print("ATTENTION_LONG_STABILITY cycle=\(context.report.completedCycleCount) generations=\(context.report.completedGenerationCount) cancelled=\(context.report.completedCancellationCount) elapsed=\(Int(context.elapsed)) thermal=\(thermalLabel())")
                // Keep a useful workload while leaving brief recovery intervals.
                // Serious thermal state is recorded and receives a longer rest.
                let rest = ProcessInfo.processInfo.thermalState == .serious ? 5.0 : 2.0
                context.report.recoveryIntervalSeconds += rest
                try await Task.sleep(for: .seconds(rest))
            }
            try context.checkpoint()
            guard context.report.completedCycleCount >= 10,
                  context.report.completedGenerationCount >= 30,
                  context.report.completedCancellationCount >= 2,
                  context.report.completedNearLimitGenerationCount >= 10,
                  context.report.completed64TokenRequests >= 10 else {
                throw LongRunFailure(code: "target-duration-reached-with-insufficient-completed-work")
            }
            context.report.status = "passed"
            #endif
        } catch is CancellationError {
            if context.report.status == "running" {
                context.stop(status: "cancelled", code: "runner-task-cancelled")
            }
        } catch let failure as LongRunFailure {
            if context.report.status == "running" {
                context.stop(status: "failed", code: failure.code)
            }
        } catch let failure as CheckFailure {
            if context.report.status == "running" {
                context.stop(status: "failed", code: failure.rawValue)
            }
        } catch {
            if context.report.status == "running" {
                context.stop(status: "failed", code: "setup-or-report-io-error")
            }
        }

        // Persist the terminal status before cleanup can suspend in background.
        context.report.phase = "cleanup"
        context.save()
        heartbeat.cancel()
        watchdog.cancel()
        #if !targetEnvironment(simulator)
        let cleanup = Task { @MainActor in
            await engine.unload()
            // A simultaneous scene-phase unload may already own cleanup.
            for _ in 0..<50 where engine.isWorking {
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
        await cleanup.value
        #endif
        context.checkEnvironment()
        if Task.isCancelled, context.report.status == "passed" {
            context.stop(status: "cancelled", code: "runner-task-cancelled-during-cleanup")
        }
        if context.report.status == "passed", !Task.isCancelled,
           UIApplication.shared.applicationState == .active, !engine.isWorking {
            do { try await context.sampleMilestone(phase: "final-unloaded") }
            catch {
                if context.report.status == "passed" {
                    context.stop(status: "failed", code: "final-foreground-memory-snapshot-unavailable")
                }
                context.sample(phase: "final-unloaded")
            }
        } else {
            // Interrupted/background cleanup never submits a diagnostic barrier.
            context.sample(phase: "final-unloaded")
        }
        if context.report.status == "passed", engine.isReady || engine.isWorking {
            context.stop(status: "failed", code: "final-unload-did-not-clear-state")
        }
        context.report.phase = "finished"
        context.report.finishedAt = stamp()
        context.report.idleTimerRestoredOnReturn = true
        context.save()
        print("ATTENTION_LONG_STABILITY status=\(context.report.status) cycles=\(context.report.completedCycleCount) generations=\(context.report.completedGenerationCount) elapsed=\(Int(context.elapsed)) failure=\(context.report.failureCode ?? "none")")
    }

    static func run(engine: LocalModelEngine) async {
        guard !hasRun else { return }
        hasRun = true
        let previousIdleTimerSetting = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = previousIdleTimerSetting }
        let started = Date()
        var report = StabilityReport(startedAt: stamp(), requestedDeadlineSeconds: deadlineSeconds)
        let deadline = DeadlineFlag()
        let watchdog = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(deadlineSeconds))
                deadline.expired = true
                engine.cancel()
            } catch { /* Cancelled after normal completion. */ }
        }
        defer { watchdog.cancel() }

        do {
            #if targetEnvironment(simulator)
            throw CheckFailure.unsupportedSimulator
            #else
            await engine.unload()
            try checkpoint(deadline)
            await appendSample(&report, engine: engine, phase: "baseline", caseID: "baseline", started: started)
            guard let baselineActive = report.samples.last?.mlxActiveBytes else {
                throw CheckFailure.memorySnapshotUnavailable
            }
            try persist(report)

            report.originalTransformerBenchmark = try benchmarkOriginalTransformer(deadline: deadline)
            try persist(report)

            // Release this temporary tokenizer before memory/inference cycles. The
            // engine owns its own tokenizer only while the model is loaded.
            let fixtures = try await buildFixtures()
            let defaultCount = fixtures.defaultCount
            guard defaultCount == 24 else { throw CheckFailure.defaultFixtureTokenCount }
            let nearLimit = fixtures.nearLimit
            try checkpoint(deadline)

            for (index, selection) in [(17, 0), (0, 7), (35, 31)].enumerated() {
                let caseID = "load-inspect-unload-\(index + 1)"
                let result = try await inferenceCycle(engine: engine, prompt: publicPrompt,
                    fixtureID: "public-pronoun-fixture", expectedCount: defaultCount,
                    layer: selection.0, head: selection.1, caseID: caseID,
                    baselineActive: baselineActive, deadline: deadline, report: &report, started: started)
                report.cases.append(result)
                try persist(report)
            }

            let nearResult = try await inferenceCycle(engine: engine, prompt: nearLimit.prompt,
                fixtureID: "public-near-limit-fixture", expectedCount: nearLimit.count,
                layer: 18, head: 15, caseID: "near-limit-inspect-unload",
                baselineActive: baselineActive, deadline: deadline, report: &report, started: started)
            report.cases.append(nearResult)
            try persist(report)

            let cancellation = try await cancellationCycle(engine: engine, expectedCount: defaultCount,
                baselineActive: baselineActive, deadline: deadline, report: &report, started: started)
            report.cases.append(cancellation)
            try checkpoint(deadline)
            report.status = "passed"
            #endif
        } catch is CancellationError {
            report.status = "cancelled"
            report.failureCode = "runner-task-cancelled"
        } catch let failure as CheckFailure {
            report.status = deadline.expired ? "timed_out" : "failed"
            report.failureCode = failure.rawValue
        } catch {
            report.status = "failed"
            report.failureCode = "setup-or-report-io-error"
        }

        await engine.unload()
        await appendSample(&report, engine: engine, phase: "final-unloaded", caseID: "cleanup", started: started)
        report.finishedAt = stamp()
        report.elapsedSeconds = Date().timeIntervalSince(started)
        report.deadlineReached = deadline.expired
        report.completedCaseCount = report.cases.count
        report.maximumSampledProcessFootprintBytes = report.samples.compactMap(\.processPhysFootprintBytes).max()
        report.maximumReportedProcessPeakBytes = report.samples.compactMap(\.processLedgerPeakFootprintBytes).max()
        report.maximumReportedMLXPeakBytes = report.samples.compactMap(\.mlxPeakBytes).max()
        do { try persist(report) }
        catch { print("ATTENTION_STABILITY_REPORT_WRITE_FAILED") }
        print("ATTENTION_STABILITY status=\(report.status) cases=\(report.completedCaseCount) elapsed=\(report.elapsedSeconds ?? 0) failure=\(report.failureCode ?? "none")")
    }

    private struct LongFixtures {
        let defaultCount: Int
        let nearLimit: (prompt: String, count: Int)
        let counting: (prompt: String, count: Int)
    }

    private static func buildLongFixtures() async throws -> LongFixtures {
        let tokenizer = try await AutoTokenizer.from(modelFolder: LocalModelEngine.modelDirectory)
        let counting = "List the integers from 1 through 50, separated by commas, with no explanation."
        return LongFixtures(defaultCount: tokenCount(publicPrompt, tokenizer: tokenizer),
            nearLimit: try nearLimitFixture(tokenizer: tokenizer),
            counting: (counting, tokenCount(counting, tokenizer: tokenizer)))
    }

    private static func longCycle(engine: LocalModelEngine, fixtures: LongFixtures,
                                  context: LongRunContext) async throws {
        let index = context.report.completedCycleCount
        let cycleID = "cycle-\(index + 1)"
        let cycleStart = context.elapsed
        context.report.currentCycleID = cycleID
        context.report.phase = "before-load"
        try await context.sampleMilestone(phase: "before-load")
        context.save()
        let loadStart = context.elapsed
        await engine.load()
        try context.checkpoint()
        guard engine.isReady, !engine.isWorking, engine.error == nil else {
            throw LongRunFailure(code: "model-not-ready-after-load")
        }
        context.report.completedLoadCount += 1
        let loadSeconds = context.elapsed - loadStart
        context.report.phase = "loaded"
        try await context.sampleMilestone(phase: "loaded")
        context.save()

        let requests: [(String, String, Int, Int, Int, Int)] = [
            ("public-pronoun-fixture", publicPrompt, fixtures.defaultCount, index % 36, (index * 5) % 32, 32),
            ("public-near-limit-fixture", fixtures.nearLimit.prompt, fixtures.nearLimit.count,
             (index * 11 + 17) % 36, (index * 7 + 15) % 32, 64),
            ("public-counting-fixture", fixtures.counting.prompt, fixtures.counting.count,
             (index * 13 + 35) % 36, (index * 3 + 31) % 32, 64)
        ]
        for (offset, request) in requests.enumerated() {
            try context.checkpoint()
            let (fixtureID, prompt, expectedCount, layer, head, limit) = request
            let operationID = "\(cycleID)-generation-\(offset + 1)"
            context.report.phase = "\(operationID)-running"
            context.save()
            let generationStart = context.elapsed
            await engine.inspect(prompt: prompt, layer: layer, head: head, maxNewTokens: limit)
            let generationSeconds = context.elapsed - generationStart
            try context.checkpoint()
            try await context.sampleMilestone(phase: "\(operationID)-completed")
            guard engine.error == nil, !engine.isWorking, engine.isReady,
                  let trace = engine.trace, trace.mode == "live" else {
                throw LongRunFailure(code: "live-trace-not-produced")
            }
            guard trace.layer == layer, trace.head == head, trace.tokens.count == expectedCount,
                  trace.model.id == LocalModelMetadata.qwen.id, trace.model.revision == LocalModelMetadata.qwen.revision,
                  trace.model.layers == 36, trace.model.heads == 32, trace.model.kvHeads == 8,
                  trace.model.headDimension == 128 else {
                throw LongRunFailure(code: "model-or-token-count-mismatch")
            }
            let validation = try validate(trace)
            guard !trace.generatedTokenIDs.isEmpty, trace.generatedTokenIDs.count <= limit,
                  !trace.nextTokens.isEmpty,
                  trace.nextTokens.allSatisfy({ $0.probability.isFinite && $0.probability >= 0 && $0.probability <= 1 }),
                  trace.nextTokens.reduce(0, { $0 + $1.probability }) <= 1.00001 else {
                throw LongRunFailure(code: "generation-or-next-token-probability-bound")
            }
            context.report.weightSHA256 = trace.provenance.weightSHA256
            context.report.runtime = trace.provenance.runtime
            context.report.operations.append(LongStabilityOperation(id: operationID, cycleID: cycleID,
                kind: "generation", fixtureID: fixtureID, layer: layer, head: head,
                inputTokenCount: trace.tokens.count, requestedMaxNewTokens: limit,
                generatedTokenCount: trace.generatedTokenIDs.count, elapsedSeconds: generationSeconds,
                rowSumMaxError: validation.rowError, maskedFutureMax: validation.futureMax,
                cancellationObserved: nil, priorTracePreserved: nil))
            context.report.completedGenerationCount += 1
            context.report.totalGeneratedTokenCount += trace.generatedTokenIDs.count
            if limit == 64 { context.report.completed64TokenRequests += 1 }
            if fixtureID == "public-near-limit-fixture" { context.report.completedNearLimitGenerationCount += 1 }
            context.save()
        }

        if (index + 1).isMultiple(of: 3) {
            try await longCancellation(engine: engine, fixtures: fixtures, cycleID: cycleID, context: context)
        }
        try context.checkpoint()
        context.report.phase = "unloading"
        context.save()
        let unloadStart = context.elapsed
        await engine.unload()
        try await Task.sleep(for: .seconds(1))
        try context.checkpoint()
        try await context.sampleMilestone(phase: "after-unload")
        guard !engine.isReady, !engine.isWorking else {
            throw LongRunFailure(code: "unload-did-not-clear-state")
        }
        let unloadSeconds = context.elapsed - unloadStart
        try context.validateUnloadMemory()
        context.report.completedCycleCount += 1
        context.report.cycles.append(LongStabilityCycle(id: cycleID, loadSeconds: loadSeconds,
            unloadAndSettleSeconds: unloadSeconds, cycleSeconds: context.elapsed - cycleStart,
            operationsCompleted: (index + 1).isMultiple(of: 3) ? 4 : 3,
            thermalStateAtCompletion: thermalLabel()))
        context.report.phase = "cycle-completed"
        context.save()
    }

    private static func longCancellation(engine: LocalModelEngine, fixtures: LongFixtures,
                                         cycleID: String, context: LongRunContext) async throws {
        try context.checkpoint()
        let previousCapture = engine.trace?.capturedAt
        let previousLayer = engine.trace?.layer
        let previousHead = engine.trace?.head
        let layer = ((previousLayer ?? 17) + 1) % 36
        let head = ((previousHead ?? 3) + 1) % 32
        let operationID = "\(cycleID)-cancellation"
        context.report.phase = "\(operationID)-running"
        context.save()
        let inspection = Task { @MainActor in
            await engine.inspect(prompt: fixtures.counting.prompt, layer: layer, head: head, maxNewTokens: 64)
        }
        try await Task.sleep(for: .milliseconds(125))
        try context.checkpoint()
        let wasWorking = engine.isWorking
        let cancelStart = context.elapsed
        engine.cancel()
        await inspection.value
        try context.checkpoint()
        let observed = engine.status == "Experiment cancelled" && !engine.isWorking
        let preserved = engine.trace?.capturedAt == previousCapture
            && engine.trace?.layer == previousLayer && engine.trace?.head == previousHead
        try await context.sampleMilestone(phase: "after-cancel")
        guard wasWorking, observed, preserved, engine.isReady else {
            throw LongRunFailure(code: "active-inspection-cancellation-check-failed")
        }
        context.report.operations.append(LongStabilityOperation(id: operationID, cycleID: cycleID,
            kind: "cancellation", fixtureID: "public-counting-fixture", layer: layer, head: head,
            inputTokenCount: fixtures.counting.count, requestedMaxNewTokens: 64, generatedTokenCount: nil,
            elapsedSeconds: context.elapsed - cancelStart, rowSumMaxError: nil, maskedFutureMax: nil,
            cancellationObserved: observed, priorTracePreserved: preserved))
        context.report.completedCancellationCount += 1
        context.save()
    }

    private static func thermalLabel() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    private static func inferenceCycle(
        engine: LocalModelEngine, prompt: String, fixtureID: String, expectedCount: Int,
        layer: Int, head: Int, caseID: String, baselineActive: Int,
        deadline: DeadlineFlag, report: inout StabilityReport, started: Date
    ) async throws -> StabilityCase {
        try checkpoint(deadline)
        await appendSample(&report, engine: engine, phase: "before-load", caseID: caseID, started: started)
        let loadStart = Date()
        await engine.load()
        let loadSeconds = Date().timeIntervalSince(loadStart)
        try checkpoint(deadline)
        await appendSample(&report, engine: engine, phase: "loaded", caseID: caseID, started: started)
        try persist(report)
        guard engine.isReady, !engine.isWorking, engine.error == nil else { throw CheckFailure.loadNotReady }

        let inspectStart = Date()
        await engine.inspect(prompt: prompt, layer: layer, head: head, maxNewTokens: 16)
        let inspectSeconds = Date().timeIntervalSince(inspectStart)
        try checkpoint(deadline)
        await appendSample(&report, engine: engine, phase: "after-capture", caseID: caseID, started: started)
        try persist(report)
        guard engine.error == nil, let trace = engine.trace, trace.mode == "live" else { throw CheckFailure.noLiveTrace }
        report.weightSHA256 = trace.provenance.weightSHA256
        report.runtime = trace.provenance.runtime
        guard trace.layer == layer, trace.head == head, trace.tokens.count == expectedCount,
              trace.model.layers == 36, trace.model.heads == 32, trace.model.kvHeads == 8,
              trace.model.headDimension == 128 else { throw CheckFailure.modelOrTokenShape }
        let validation = try validate(trace)
        guard trace.generatedTokenIDs.count > 0, trace.generatedTokenIDs.count <= 16 else { throw CheckFailure.generationBound }

        let unloadStart = Date()
        await engine.unload()
        try await Task.sleep(for: .milliseconds(120))
        let unloadSeconds = Date().timeIntervalSince(unloadStart)
        await appendSample(&report, engine: engine, phase: "after-unload", caseID: caseID, started: started)
        try persist(report)
        guard !engine.isReady, !engine.isWorking else { throw CheckFailure.unloadState }
        guard let active = report.samples.last?.mlxActiveBytes else {
            throw CheckFailure.memorySnapshotUnavailable
        }
        guard active <= baselineActive + 16 * 1024 * 1024 else {
            throw CheckFailure.mlxAllocationsRetained
        }
        return StabilityCase(id: caseID, fixtureID: fixtureID, layer: layer, head: head,
            inputTokenCount: trace.tokens.count, generatedTokenCount: trace.generatedTokenIDs.count,
            status: "passed", timings: ["loadSeconds": loadSeconds, "captureAndGenerationSeconds": inspectSeconds,
                                       "unloadAndSettleSeconds": unloadSeconds],
            checks: ["actualLiveInference": true, "modelDimensionsMatch": true, "inputTokenCountMatches": true,
                     "matrixFiniteNonnegative": true, "rowNormalization": true, "causalMask": true,
                     "generationBound": true, "readyClearedAfterUnload": true, "workingClearedAfterUnload": true,
                     "mlxActiveNearBaselineAfterUnload": true],
            rowSumMaxError: validation.rowError, maskedFutureMax: validation.futureMax,
            cancellationRequested: false, cancellationObserved: nil, priorTracePreserved: nil)
    }

    private static func cancellationCycle(
        engine: LocalModelEngine, expectedCount: Int, baselineActive: Int,
        deadline: DeadlineFlag, report: inout StabilityReport, started: Date
    ) async throws -> StabilityCase {
        let caseID = "cancel-active-inspection-unload"
        try checkpoint(deadline)
        let loadStart = Date()
        await engine.load()
        let loadSeconds = Date().timeIntervalSince(loadStart)
        guard engine.isReady else { throw CheckFailure.loadNotReady }
        await appendSample(&report, engine: engine, phase: "loaded-before-cancel", caseID: caseID, started: started)
        try persist(report)
        let previousCapture = engine.trace?.capturedAt
        let previousLayer = engine.trace?.layer
        let previousHead = engine.trace?.head
        let inspectionStart = Date()
        let inspection = Task { @MainActor in
            await engine.inspect(prompt: publicPrompt, layer: 17, head: 3, maxNewTokens: 64)
        }
        try await Task.sleep(for: .milliseconds(75))
        let wasWorking = engine.isWorking
        let cancelRequest = Date()
        engine.cancel()
        await inspection.value
        let cancelSeconds = Date().timeIntervalSince(cancelRequest)
        let inspectionSeconds = Date().timeIntervalSince(inspectionStart)
        let cancellationObserved = engine.status == "Experiment cancelled" && !engine.isWorking
        let priorTracePreserved = engine.trace?.capturedAt == previousCapture
            && engine.trace?.layer == previousLayer && engine.trace?.head == previousHead
        await appendSample(&report, engine: engine, phase: "after-cancel", caseID: caseID, started: started)
        try persist(report)
        await engine.unload()
        try await Task.sleep(for: .milliseconds(120))
        await appendSample(&report, engine: engine, phase: "after-unload", caseID: caseID, started: started)
        try persist(report)
        try checkpoint(deadline)
        guard wasWorking, cancellationObserved, priorTracePreserved, !engine.isReady, !engine.isWorking else {
            throw CheckFailure.cancellationNotObserved
        }
        guard let active = report.samples.last?.mlxActiveBytes else {
            throw CheckFailure.memorySnapshotUnavailable
        }
        guard active <= baselineActive + 16 * 1024 * 1024 else {
            throw CheckFailure.mlxAllocationsRetained
        }
        return StabilityCase(id: caseID, fixtureID: "public-pronoun-fixture", layer: 17, head: 3,
            inputTokenCount: expectedCount, generatedTokenCount: nil, status: "passed",
            timings: ["loadSeconds": loadSeconds, "cancelRequestToCompletionSeconds": cancelSeconds,
                      "inspectionStartToCancelledCompletionSeconds": inspectionSeconds],
            checks: ["operationWasActiveAtCancel": wasWorking, "cancellationObserved": cancellationObserved,
                     "priorTracePreserved": priorTracePreserved, "readyClearedAfterUnload": true,
                     "workingClearedAfterUnload": true, "mlxActiveNearBaselineAfterUnload": true],
            rowSumMaxError: nil, maskedFutureMax: nil, cancellationRequested: true,
            cancellationObserved: cancellationObserved, priorTracePreserved: priorTracePreserved)
    }

    private static func validate(_ trace: LocalModelTrace) throws -> (rowError: Double, futureMax: Double) {
        let n = trace.tokens.count
        guard n > 0, n <= 96, trace.attention.count == n,
              trace.attention.allSatisfy({ $0.count == n }) else { throw CheckFailure.matrixShape }
        guard trace.attention.allSatisfy({ $0.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 } }) else {
            throw CheckFailure.invalidProbability
        }
        let rowError = trace.attention.map { abs($0.reduce(0, +) - 1) }.max() ?? 0
        let futureMax = trace.attention.enumerated().flatMap { index, row in row.dropFirst(index + 1) }.max() ?? 0
        guard rowError <= 1e-5, futureMax == 0,
              abs(rowError - trace.metrics.rowSumMaxError) <= 1e-12,
              futureMax == trace.metrics.maskedFutureMax else { throw CheckFailure.attentionChecks }
        return (rowError, futureMax)
    }

    /// Development benchmark deliberately performs synchronous CPU forward calls.
    /// Normal machine work runs off the UI thread; this does not measure frame latency.
    private static func benchmarkOriginalTransformer(deadline: DeadlineFlag) throws -> OriginalTransformerBenchmark {
        let source = Array(repeating: ["attention", "connects", "tokens", "from", "context", "light"], count: 4).flatMap { $0 }
        let target = Array(source.prefix(23))
        let transformer = TransformerEngine()
        var elapsed: [Double] = []
        var observedChecksum = 0.0
        for index in 0..<10 {
            try checkpoint(deadline)
            var controls = TransformerControls(headCount: 8, layerCount: 6)
            controls.attentionTemperature = 0.8 + Double(index) * 0.04
            let callStart = Date()
            let trace = transformer.forward(sourceTokens: source, targetTokens: target, controls: controls)
            elapsed.append(Date().timeIntervalSince(callStart))
            guard trace.sourceTokens.count == 24, trace.decoderTokens.count == 24,
                  trace.encoderLayers.count == 6, trace.decoderLayers.count == 6,
                  trace.encoderLayers.allSatisfy({ $0.selfAttention.heads.count == 8 }),
                  trace.decoderLayers.allSatisfy({ $0.selfAttention.heads.count == 8 && $0.crossAttention.heads.count == 8 })
            else { throw CheckFailure.originalTransformerShape }
            observedChecksum += trace.probabilities.row(23)[index % trace.probabilities.columns]
        }
        guard observedChecksum.isFinite else { throw CheckFailure.originalTransformerShape }
        return OriginalTransformerBenchmark(fixtureID: "public-maximum-shape-teaching-transformer",
            executionContext: "synchronous-main-actor", modelDimension: TransformerEngine.dimension,
            sourceTokenCount: source.count, suppliedTargetTokenCount: target.count, decoderInputTokenCount: 24,
            encoderLayers: 6, decoderLayers: 6, headCount: 8, completedForwardCalls: elapsed.count,
            perCallSeconds: elapsed, averageSeconds: elapsed.reduce(0, +) / Double(elapsed.count),
            maximumSeconds: elapsed.max() ?? 0, outputChecksum: observedChecksum)
    }

    private static func tokenCount(_ prompt: String, tokenizer: any Tokenizers.Tokenizer) -> Int {
        let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer.encode(text: formatted, addSpecialTokens: false).count
    }

    private static func buildFixtures() async throws -> (defaultCount: Int, nearLimit: (prompt: String, count: Int)) {
        let tokenizer = try await AutoTokenizer.from(modelFolder: LocalModelEngine.modelDirectory)
        return (tokenCount(publicPrompt, tokenizer: tokenizer), try nearLimitFixture(tokenizer: tokenizer))
    }

    private static func nearLimitFixture(tokenizer: any Tokenizers.Tokenizer) throws -> (prompt: String, count: Int) {
        var prompt = publicPrompt
        var count = tokenCount(prompt, tokenizer: tokenizer)
        for _ in 0..<96 {
            let candidate = prompt + " attention"
            let candidateCount = tokenCount(candidate, tokenizer: tokenizer)
            if candidateCount > 95 { break }
            prompt = candidate
            count = candidateCount
        }
        guard (90...96).contains(count) else { throw CheckFailure.nearLimitFixture }
        return (prompt, count)
    }

    private static func checkpoint(_ deadline: DeadlineFlag) throws {
        try Task.checkCancellation()
        if deadline.expired { throw CheckFailure.deadline }
    }

    private static func appendSample(_ report: inout StabilityReport, engine: LocalModelEngine,
                                     phase: String, caseID: String, started: Date) async {
        var snapshot: Memory.Snapshot?
        #if DEBUG && !targetEnvironment(simulator)
        if UIApplication.shared.applicationState == .active, !Task.isCancelled, !engine.isWorking {
            snapshot = await engine.stabilityMemorySnapshot()
        }
        #endif
        // A scene transition or cancellation during the actor hop invalidates
        // the diagnostic sample. Mach sampling does not submit GPU work.
        if UIApplication.shared.applicationState != .active || Task.isCancelled || engine.isWorking {
            snapshot = nil
        }
        report.samples.append(memorySample(engine: engine, phase: phase, caseID: caseID,
                                          started: started, mlxSnapshot: snapshot))
    }

    private static func memorySample(engine: LocalModelEngine, phase: String, caseID: String,
                                     started: Date, mlxSnapshot: Memory.Snapshot? = nil,
                                     elapsedSeconds: Double? = nil) -> StabilityMemorySample {
        var info = task_vm_info_data_t()
        let capacity = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        var count = capacity
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(capacity)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        #if targetEnvironment(simulator)
        let mlxActive: Int? = nil
        let mlxCache: Int? = nil
        let mlxPeak: Int? = nil
        #else
        let mlx = mlxSnapshot
        let mlxActive: Int? = mlx?.activeMemory
        let mlxCache: Int? = mlx?.cacheMemory
        let mlxPeak: Int? = mlx?.peakMemory
        #endif
        return StabilityMemorySample(caseID: caseID, phase: phase, capturedAt: stamp(),
            elapsedSeconds: elapsedSeconds ?? Date().timeIntervalSince(started), machStatus: result,
            processPhysFootprintBytes: result == KERN_SUCCESS ? info.phys_footprint : nil,
            processLedgerPeakFootprintBytes: result == KERN_SUCCESS && info.ledger_phys_footprint_peak > 0 ? UInt64(info.ledger_phys_footprint_peak) : nil,
            mlxActiveBytes: mlxActive, mlxCacheBytes: mlxCache, mlxPeakBytes: mlxPeak,
            engineReady: engine.isReady, engineWorking: engine.isWorking)
    }

    private static func persist(_ report: StabilityReport) throws {
        let documents = URL.documentsDirectory
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: documents.appendingPathComponent("StabilityValidation.json"), options: .atomic)
    }

    private static func stamp() -> String { ISO8601DateFormatter().string(from: Date()) }

    @MainActor
    private final class LongRunContext {
        let engine: LocalModelEngine
        let started = Date()
        let startedUptime = ProcessInfo.processInfo.systemUptime
        var report = LongStabilityReport(startedAt: DeviceStabilityCheck.stamp())
        var foregroundEstablished = false

        init(engine: LocalModelEngine) { self.engine = engine }
        var elapsed: Double { ProcessInfo.processInfo.systemUptime - startedUptime }

        func stop(status: String, code: String) {
            guard report.status == "running" || report.status == "passed" else { return }
            report.status = status
            report.failureCode = code
            report.interruptionElapsedSeconds = elapsed
            // Save before cancelling work or awaiting scene cleanup.
            save()
            engine.cancel()
        }

        func checkEnvironment() {
            guard report.status == "running" || report.status == "passed" else { return }
            if elapsed >= report.cooperativeDeadlineSeconds {
                stop(status: "timed_out", code: "cooperative-deadline-reached")
            } else if foregroundEstablished, UIApplication.shared.applicationState != .active {
                stop(status: "interrupted", code: "foreground-interrupted")
            } else if ProcessInfo.processInfo.thermalState == .critical {
                stop(status: "aborted", code: "critical-thermal-state")
            }
        }

        func checkpoint() throws {
            checkEnvironment()
            try Task.checkCancellation()
            guard report.status == "running" else {
                throw LongRunFailure(code: report.failureCode ?? "run-no-longer-active")
            }
        }

        func sampleMilestone(phase: String) async throws {
            checkEnvironment()
            try Task.checkCancellation()
            guard report.status == "running" || report.status == "passed" else {
                throw LongRunFailure(code: report.failureCode ?? "run-no-longer-active")
            }
            guard foregroundEstablished, UIApplication.shared.applicationState == .active,
                  !engine.isWorking else {
                throw LongRunFailure(code: "foreground-idle-memory-snapshot-unavailable")
            }
            #if DEBUG && !targetEnvironment(simulator)
            let snapshot = await engine.stabilityMemorySnapshot()
            #else
            let snapshot: Memory.Snapshot? = nil
            #endif
            checkEnvironment()
            try Task.checkCancellation()
            guard report.status == "running" || report.status == "passed" else {
                throw LongRunFailure(code: report.failureCode ?? "run-no-longer-active")
            }
            guard UIApplication.shared.applicationState == .active, !engine.isWorking,
                  let snapshot else {
                throw LongRunFailure(code: "foreground-synchronized-memory-snapshot-unavailable")
            }
            sample(phase: phase, mlxSnapshot: snapshot)
        }

        func sample(phase: String, mlxSnapshot: Memory.Snapshot? = nil) {
            let memory = DeviceStabilityCheck.memorySample(engine: engine, phase: phase,
                caseID: report.currentCycleID ?? "setup", started: started,
                mlxSnapshot: mlxSnapshot, elapsedSeconds: elapsed)
            report.samples.append(LongMemorySample(memory: memory,
                thermalState: DeviceStabilityCheck.thermalLabel(),
                applicationState: applicationStateLabel(),
                lastObservedPressureEvent: report.pressureEvents.last?.level ?? "none-observed",
                quiescentMLXSample: mlxSnapshot != nil))
            if let footprint = memory.processPhysFootprintBytes,
               footprint > report.priorMeasuredProcessPeakBytes + report.loadedProcessAllowanceBytes {
                stop(status: "failed", code: "process-footprint-above-prior-peak-plus-development-allowance")
            }
        }

        func validateUnloadMemory() throws {
            guard let last = report.samples.last, last.quiescentMLXSample,
                  let active = last.memory.mlxActiveBytes, let cache = last.memory.mlxCacheBytes,
                  let footprint = last.memory.processPhysFootprintBytes else {
                throw LongRunFailure(code: "quiescent-unload-memory-measurement-unavailable")
            }
            guard cache == 0 else { throw LongRunFailure(code: "mlx-cache-nonzero-after-unload") }
            let baseline = report.baselineMLXActiveBytes ?? 0
            let allowance = baseline + report.completedLoadCount * report.knownMLXResidualAllowancePerLoadBytes
                + report.additionalMLXResidualAllowanceBytes
            report.unloadedMeasurements.append(LongUnloadedMeasurement(loadCount: report.completedLoadCount,
                cycleID: report.currentCycleID ?? "unknown", elapsedSeconds: elapsed,
                processPhysFootprintBytes: footprint, mlxActiveBytes: active, mlxCacheBytes: cache,
                mlxGrowthFromBaselineBytes: active - baseline, mlxActiveGuardBytes: allowance))
            updateTrends()
            guard active <= allowance else {
                throw LongRunFailure(code: "mlx-residual-above-known-per-load-trend-plus-development-allowance")
            }
            if report.unloadedMeasurements.count >= 6,
               let median = report.lastThreeUnloadedFootprintMedianBytes,
               let warmup = report.firstThreeUnloadedFootprintMedianBytes,
               median > warmup + report.unloadedProcessAllowanceBytes {
                throw LongRunFailure(code: "last-three-unloaded-footprint-median-above-warmup-plus128MiB")
            }
        }

        private func updateTrends() {
            let samples = report.unloadedMeasurements
            if samples.count >= 3 {
                report.firstThreeUnloadedFootprintMedianBytes = median(Array(samples.prefix(3)).map(\.processPhysFootprintBytes))
                report.lastThreeUnloadedFootprintMedianBytes = median(Array(samples.suffix(3)).map(\.processPhysFootprintBytes))
                if let first = report.firstThreeUnloadedFootprintMedianBytes,
                   let last = report.lastThreeUnloadedFootprintMedianBytes {
                    report.unloadedFootprintMedianChangeBytes = Double(last) - Double(first)
                }
            }
            report.unloadedFootprintLinearTrendBytesPerLoad = slope(samples.map { Double($0.loadCount) },
                samples.map { Double($0.processPhysFootprintBytes) })
            report.unloadedMLXLinearTrendBytesPerLoad = slope(samples.map { Double($0.loadCount) },
                samples.map { Double($0.mlxActiveBytes) })
            report.finalObservedMLXGrowthBytes = samples.last?.mlxGrowthFromBaselineBytes
        }

        func save() {
            report.updatedAt = DeviceStabilityCheck.stamp()
            report.elapsedSeconds = elapsed
            report.maximumSampledProcessFootprintBytes = report.samples.compactMap(\.memory.processPhysFootprintBytes).max()
            report.maximumReportedProcessPeakBytes = report.samples.compactMap(\.memory.processLedgerPeakFootprintBytes).max()
            report.maximumReportedMLXPeakBytes = report.samples.compactMap(\.memory.mlxPeakBytes).max()
            report.maximumRowSumError = report.operations.compactMap(\.rowSumMaxError).max()
            report.maximumMaskedFutureWeight = report.operations.compactMap(\.maskedFutureMax).max()
            do {
                try FileManager.default.createDirectory(at: URL.documentsDirectory, withIntermediateDirectories: true)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(to: URL.documentsDirectory.appendingPathComponent("LongStabilityValidation.json"), options: .atomic)
            } catch {
                report.status = "failed"
                report.failureCode = "incremental-report-write-failed"
                engine.cancel()
                print("ATTENTION_LONG_STABILITY_REPORT_WRITE_FAILED")
            }
        }

        private func applicationStateLabel() -> String {
            switch UIApplication.shared.applicationState {
            case .active: "active"
            case .inactive: "inactive"
            case .background: "background"
            @unknown default: "unknown"
            }
        }

        private func median(_ values: [UInt64]) -> UInt64? {
            guard !values.isEmpty else { return nil }
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }

        private func slope(_ x: [Double], _ y: [Double]) -> Double? {
            guard x.count == y.count, x.count > 1 else { return nil }
            let meanX = x.reduce(0, +) / Double(x.count)
            let meanY = y.reduce(0, +) / Double(y.count)
            let denominator = x.reduce(0) { $0 + pow($1 - meanX, 2) }
            guard denominator > 0 else { return nil }
            return zip(x, y).reduce(0) { $0 + ($1.0 - meanX) * ($1.1 - meanY) } / denominator
        }
    }

    private struct LongRunFailure: Error { let code: String }

    private final class DeadlineFlag { var expired = false }
    private enum CheckFailure: String, Error {
        case unsupportedSimulator = "physical-device-required"
        case defaultFixtureTokenCount = "public-default-token-count-mismatch"
        case nearLimitFixture = "near-limit-fixture-not-within-safe-bound"
        case loadNotReady = "model-not-ready-after-load"
        case noLiveTrace = "live-trace-not-produced"
        case modelOrTokenShape = "model-or-token-count-mismatch"
        case generationBound = "generation-bound-check-failed"
        case unloadState = "unload-did-not-clear-state"
        case memorySnapshotUnavailable = "foreground-synchronized-memory-snapshot-unavailable"
        case mlxAllocationsRetained = "mlx-active-allocation-above-baseline-plus-16MiB-after-unload"
        case cancellationNotObserved = "active-inspection-cancellation-check-failed"
        case matrixShape = "attention-matrix-shape-mismatch"
        case invalidProbability = "attention-probability-invalid"
        case attentionChecks = "attention-normalization-or-causal-check-failed"
        case deadline = "cooperative-deadline-reached"
        case originalTransformerShape = "original-transformer-benchmark-shape-check-failed"
    }
}

private struct StabilityReport: Codable {
    var schemaVersion = 1
    var reportKind = "bounded-development-device-stability-check"
    var startedAt: String
    var requestedDeadlineSeconds: Double
    var finishedAt: String?
    var elapsedSeconds: Double?
    var status = "running"
    var failureCode: String?
    var deadlineReached = false
    var completedCaseCount = 0
    var model = LocalModelMetadata.qwen
    var weightSHA256: String?
    var runtime: String?
    var cases: [StabilityCase] = []
    var samples: [StabilityMemorySample] = []
    var maximumSampledProcessFootprintBytes: UInt64?
    var maximumReportedProcessPeakBytes: UInt64?
    var maximumReportedMLXPeakBytes: Int?
    var originalTransformerBenchmark: OriginalTransformerBenchmark?
    var limitations = ["Bounded development run; does not establish long-term stability.",
                      "Process phys_footprint samples and process ledger peak differ from MLX allocator counters.",
                      "Cancellation is cooperative; an in-flight weight load or GPU evaluation may finish before cancellation is observed.",
                      "MLX and process ledger peaks are process-lifetime high-water marks; they are not reset for this run.",
                      "The original Transformer benchmark measures synchronous forward-call wall time, not UI frames or slider latency.",
                      "No physical audio, haptic feel or accessibility claim is made."]
    var excludedContent = ["prompt text", "token IDs", "token text", "generated text", "attention matrices"]
}

private struct OriginalTransformerBenchmark: Codable {
    let fixtureID: String
    let executionContext: String
    let modelDimension: Int
    let sourceTokenCount: Int
    let suppliedTargetTokenCount: Int
    let decoderInputTokenCount: Int
    let encoderLayers: Int
    let decoderLayers: Int
    let headCount: Int
    let completedForwardCalls: Int
    let perCallSeconds: [Double]
    let averageSeconds: Double
    let maximumSeconds: Double
    let outputChecksum: Double
}

private struct StabilityCase: Codable {
    let id: String
    let fixtureID: String
    let layer: Int
    let head: Int
    let inputTokenCount: Int
    let generatedTokenCount: Int?
    let status: String
    let timings: [String: Double]
    let checks: [String: Bool]
    let rowSumMaxError: Double?
    let maskedFutureMax: Double?
    let cancellationRequested: Bool
    let cancellationObserved: Bool?
    let priorTracePreserved: Bool?
}

private struct StabilityMemorySample: Codable {
    let caseID: String
    let phase: String
    let capturedAt: String
    let elapsedSeconds: Double
    let machStatus: kern_return_t
    let processPhysFootprintBytes: UInt64?
    let processLedgerPeakFootprintBytes: UInt64?
    let mlxActiveBytes: Int?
    let mlxCacheBytes: Int?
    let mlxPeakBytes: Int?
    let engineReady: Bool
    let engineWorking: Bool
}

private struct LongStabilityReport: Codable {
    var schemaVersion = 1
    var reportKind = "twenty-minute-foreground-device-stability-check"
    var startedAt: String
    var updatedAt: String?
    var finishedAt: String?
    var elapsedSeconds = 0.0
    var targetDurationSeconds = 1_200.0
    var cooperativeDeadlineSeconds = 1_260.0
    var status = "running"
    var phase = "setup"
    var failureCode: String?
    var interruptionElapsedSeconds: Double?
    var currentCycleID: String?
    var idleTimerRestoredOnReturn = false
    var model = LocalModelMetadata.qwen
    var weightSHA256: String?
    var runtime: String?
    var completedCycleCount = 0
    var completedLoadCount = 0
    var completedGenerationCount = 0
    var completedCancellationCount = 0
    var completedNearLimitGenerationCount = 0
    var completed64TokenRequests = 0
    var totalGeneratedTokenCount = 0
    var recoveryIntervalSeconds = 0.0
    var operations: [LongStabilityOperation] = []
    var cycles: [LongStabilityCycle] = []
    var samples: [LongMemorySample] = []
    var pressureEvents: [LongPressureEvent] = []
    var unloadedMeasurements: [LongUnloadedMeasurement] = []
    var baselineMLXActiveBytes: Int?
    var maximumSampledProcessFootprintBytes: UInt64?
    var maximumReportedProcessPeakBytes: UInt64?
    var maximumReportedMLXPeakBytes: Int?
    var maximumRowSumError: Double?
    var maximumMaskedFutureWeight: Double?
    var firstThreeUnloadedFootprintMedianBytes: UInt64?
    var lastThreeUnloadedFootprintMedianBytes: UInt64?
    var unloadedFootprintMedianChangeBytes: Double?
    var unloadedFootprintLinearTrendBytesPerLoad: Double?
    var unloadedMLXLinearTrendBytesPerLoad: Double?
    var finalObservedMLXGrowthBytes: Int?
    var knownMLXResidualAllowancePerLoadBytes = 4_096
    var additionalMLXResidualAllowanceBytes = 1_048_576
    var unloadedProcessAllowanceBytes: UInt64 = 134_217_728
    var priorMeasuredProcessPeakBytes: UInt64 = 2_662_000_000
    var loadedProcessAllowanceBytes: UInt64 = 536_870_912
    var guardExplanation = [
        "After each unload, cache must equal zero and MLX active bytes must stay below initial baseline + completed loads * 4096 + 1 MiB.",
        "The previous five-load run retained about 4 KiB per load. This known nonzero growth is reported explicitly and is not declared fixed or leak-free.",
        "After six unloads, the last three process-footprint median must stay below the first three warmup median + 128 MiB.",
        "Any sampled process footprint above the prior measured 2.662 GB peak + 512 MiB aborts as a development guard failure.",
        "These measured-envelope guards detect large growth; they do not establish that all smaller allocations are necessary or harmless."
    ]
    var workload = [
        "One fresh model load per cycle, followed by three generations and an unload.",
        "Generation requests are 32 tokens for a public 24-token fixture, 64 for a public near-95-token fixture, and 64 for a public counting fixture.",
        "Layers and query heads change for every generation. Every third cycle cancels an active 64-token request and verifies preservation of the prior trace.",
        "Actual generated-token counts may be shorter than the requested bound when the model emits an end token.",
        "Recovery intervals are two seconds, or five seconds in serious thermal state. Critical thermal state or critical pressure aborts."
    ]
    var successCriteria = [
        "At least 1200 monotonic elapsed seconds without interruption or error; cooperative deadline 1260 seconds.",
        "At least ten complete load/generate/unload cycles, thirty validated generations, ten near-limit generations and ten completed 64-token requests.",
        "At least two observed active-inspection cancellations with the prior capture preserved.",
        "All completed captures have the expected model/token/matrix shape, finite probabilities, normalized rows and zero masked-future attention.",
        "Every completed unload clears ready/working state and passes the recorded allocator/process envelope guards."
    ]
    var limitations = [
        "Development foreground workload only; not an hour-long, background, thermal-throttling or battery-life guarantee.",
        "Foreground resignation interrupts the run; it never silently resumes or counts a suspended interval as passing work.",
        "Hard deadline and cancellation are cooperative; an in-flight CPU weight load or GPU evaluation may need to finish before cleanup returns.",
        "Heartbeat samples contain Mach process memory and thermal state. MLX counters are read on the model worker only at foreground milestones after a stream synchronization barrier; upstream allocator getters are not synchronized.",
        "MLX active/cache/peak fields are sequential allocator reads, not an atomic snapshot. Allocator and process-ledger peaks are process-lifetime values and are not reset.",
        "Process phys_footprint includes runtime/driver/compressed-memory accounting and is distinct from resident-page RSS and MLX allocations.",
        "Pressure levels are OS DispatchSource notifications; none-observed means no pressure notification was received, not proof of normal pressure.",
        "A process killed before final persistence leaves a running partial report; the observer must report that as incomplete.",
        "No physical audio, haptic feel, accessibility or UI-frame-latency claim is made."
    ]
    var excludedContent = ["prompt text", "token IDs", "token text", "generated text", "attention matrices", "device identifiers"]
}

private struct LongStabilityOperation: Codable {
    let id: String
    let cycleID: String
    let kind: String
    let fixtureID: String
    let layer: Int
    let head: Int
    let inputTokenCount: Int
    let requestedMaxNewTokens: Int
    let generatedTokenCount: Int?
    let elapsedSeconds: Double
    let rowSumMaxError: Double?
    let maskedFutureMax: Double?
    let cancellationObserved: Bool?
    let priorTracePreserved: Bool?
    var status = "passed"
}

private struct LongStabilityCycle: Codable {
    let id: String
    let loadSeconds: Double
    let unloadAndSettleSeconds: Double
    let cycleSeconds: Double
    let operationsCompleted: Int
    let thermalStateAtCompletion: String
    var status = "passed"
}

private struct LongMemorySample: Codable {
    let memory: StabilityMemorySample
    let thermalState: String
    let applicationState: String
    let lastObservedPressureEvent: String
    let quiescentMLXSample: Bool
}

private struct LongPressureEvent: Codable {
    let capturedAt: String
    let elapsedSeconds: Double
    let level: String
}

private struct LongUnloadedMeasurement: Codable {
    let loadCount: Int
    let cycleID: String
    let elapsedSeconds: Double
    let processPhysFootprintBytes: UInt64
    let mlxActiveBytes: Int
    let mlxCacheBytes: Int
    let mlxGrowthFromBaselineBytes: Int
    let mlxActiveGuardBytes: Int
}
