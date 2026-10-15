import XCTest

/// Short UI stress cases for the transparent machine. These never load a language model.
@MainActor
final class AttentionQualityUITests: XCTestCase {
    private var sensorySheetExpected = false
    private var largestObservedScreen = CGRect.zero
    override func setUpWithError() throws { continueAfterFailure = false }

    func testTwoCompleteStageCyclesKeepControlsResponsive() throws {
        let app = XCUIApplication()
        app.launch()
        let title = element("machine-stage-title", in: app)
        XCTAssertTrue(title.waitForExistence(timeout: 10))
        let titles = ["Embedding + position", "Multi-head attention", "Add & normalize",
                      "Feed-forward network", "Masked decoder", "Encoder → decoder", "Next-token distribution"]
        var stage = 1
        for _ in 0..<14 {
            let next = element("next-stage", in: app)
            XCTAssertTrue(next.isHittable)
            next.tap()
            stage = (stage + 1) % titles.count
            XCTAssertEqual(title.label, titles[stage])
            XCTAssertTrue(element("machine-controls", in: app).isHittable)
        }
        XCTAssertEqual(title.label, "Multi-head attention")
    }

    func testEmptyAndOversizedInputsProduceBoundedEncoderAndDecoderMatrices() throws {
        let app = XCUIApplication()
        // UserDefaults argument-domain values seed this process without changing stored sentences.
        app.launchArguments = ["-machine.source", "", "-machine.target", ""]
        app.launch()
        expectMatrix(rows: 1, columns: 1, in: app)
        for _ in 0..<3 { element("next-stage", in: app).tap() }
        XCTAssertEqual(element("machine-stage-title", in: app).label, "Masked decoder")
        expectMatrix(rows: 1, columns: 1, in: app)
        app.terminate()

        app.launchArguments = [
            "-machine.source", Array(repeating: "light", count: 26).joined(separator: " "),
            "-machine.target", Array(repeating: "we", count: 25).joined(separator: " ")
        ]
        app.launch()
        expectMatrix(rows: 24, columns: 24, in: app)
        for _ in 0..<3 { element("next-stage", in: app).tap() }
        expectMatrix(rows: 24, columns: 24, in: app)
        element("next-stage", in: app).tap()
        XCTAssertEqual(element("machine-stage-title", in: app).label, "Encoder → decoder")
        expectMatrix(rows: 24, columns: 24, in: app)
        app.terminate()
        app.launchArguments = []
    }

    func testRestoringControlsAndLeavingTheMachineStopsPlayback() throws {
        let app = XCUIApplication()
        app.launch()
        let play = element("play-machine", in: app)
        XCTAssertTrue(play.waitForExistence(timeout: 10))
        play.tap()
        XCTAssertEqual(play.label, "Pause machine")
        app.tabBars.buttons["Paper"].tap()
        app.tabBars.buttons["Machine"].tap()
        XCTAssertEqual(play.label, "Play machine")

        element("machine-controls", in: app).tap()
        let position = app.sliders["Position strength"]
        try reveal(position, in: app)
        let originalPosition = position.value as? String
        position.adjust(toNormalizedSliderPosition: 0)
        XCTAssertNotEqual(position.value as? String, originalPosition)
        let mask = app.switches["causal-mask"]
        try tapSwitch(mask, in: app)
        XCTAssertEqual(mask.value as? String, "0")
        let restore = app.buttons["Restore teaching model"]
        try reveal(restore, in: app)
        restore.tap()
        try reveal(mask, in: app, towardTop: true)
        XCTAssertEqual(mask.value as? String, "1")
        try reveal(position, in: app)
        XCTAssertEqual(position.value as? String, originalPosition)
        element("done-machine-controls", in: app).tap()

        play.tap()
        XCTAssertEqual(play.label, "Pause machine")
        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertEqual(play.label, "Play machine")
        XCTAssertTrue(element("next-stage", in: app).isHittable)
    }

    func testTallCrossAttentionFitsWithinTheMachineViewport() throws {
        let app = XCUIApplication()
        app.launchArguments = [
            "-machine.source", "hello",
            "-machine.target", Array(repeating: "we", count: 23).joined(separator: " ")
        ]
        app.launch()
        defer { app.terminate(); app.launchArguments = [] }
        XCTAssertTrue(element("next-stage", in: app).waitForExistence(timeout: 10))
        for _ in 0..<4 { element("next-stage", in: app).tap() }
        XCTAssertEqual(element("machine-stage-title", in: app).label, "Encoder → decoder")
        expectMatrix(rows: 24, columns: 1, in: app)
        let matrix = element("attention-matrix", in: app)
        XCTAssertGreaterThan(matrix.frame.height, 0)
        XCTAssertLessThanOrEqual(matrix.frame.height, 260)
        XCTAssertTrue(element("next-stage", in: app).isHittable)
    }

    func testSensoryPreferencesGateFeedbackAndStopPreviewWithoutChangingSavedPreferences() throws {
        // Keep cleanup reachable after an assertion failure in this preference-changing case.
        continueAfterFailure = true
        let app = XCUIApplication()
        app.launchArguments = []
        app.launch()
        try openSensoryCheck(in: app)

        let sound = app.switches["sound-toggle"]
        let haptics = app.switches["haptics-toggle"]
        let volume = app.sliders["sensory-volume"]
        // Capture the current saved settings before any preference-changing control interaction.
        let savedSound = try XCTUnwrap(sound.value as? String)
        let savedHaptics = try XCTUnwrap(haptics.value as? String)
        try reveal(volume, in: app)
        let savedVolume = try normalizedVolume(volume)

        defer {
            app.terminate()
            app.launchArguments = []
            app.launch()
            if (try? openSensoryCheck(in: app)) != nil {
                do {
                    try restoreSwitch(app.switches["sound-toggle"], to: savedSound, in: app)
                    try restoreSwitch(app.switches["haptics-toggle"], to: savedHaptics, in: app)
                    try reveal(app.sliders["sensory-volume"], in: app, towardTop: true)
                    app.sliders["sensory-volume"].adjust(toNormalizedSliderPosition: savedVolume)
                    try tap("sensory-stop", in: app)
                    try tap("sensory-done", in: app)
                } catch { XCTFail("Could not restore saved sensory preferences: \(error)") }
            }
        }

        try restoreSwitch(sound, to: "0", in: app)
        try restoreSwitch(haptics, to: "1", in: app)
        try reveal(volume, in: app)
        volume.adjust(toNormalizedSliderPosition: 0.5)
        XCTAssertEqual(sound.value as? String, "0")
        XCTAssertEqual(haptics.value as? String, "1")
        XCTAssertEqual(try normalizedVolume(volume), 0.5, accuracy: 0.03)

        try tapSwitch(sound, in: app)
        XCTAssertEqual(sound.value as? String, "1")
        try tap("sensory-strong", in: app)
        try expectStatus("Tone requested", in: app)
        try tap("sensory-stop", in: app, towardTop: true)
        try expectStatus("Stopped", in: app)
        assertNoPlayingVoicesIfExposed(in: app)

        // Begin real feedback before leaving; a suspended sequence must never resume on return.
        let sequence = element("sensory-sequence", in: app)
        try reveal(sequence, in: app, towardTop: true)
        sequence.tap()
        XCTAssertFalse(sequence.isEnabled)
        XCUIDevice.shared.press(.home)
        app.activate()
        guard sequence.waitForExistence(timeout: 5) else {
            XCTFail("The feedback sheet disappeared after returning from Home")
            throw NSError(domain: "AttentionQualityUITests", code: 2)
        }
        XCTAssertTrue(sequence.isEnabled)
        try expectStatus("Stopped", in: app)
        try expectStoppedFeedbackRemainsFrozen(in: app)

        try reveal(volume, in: app, towardTop: true)
        volume.adjust(toNormalizedSliderPosition: 0)
        XCTAssertEqual(try normalizedVolume(volume), 0, accuracy: 0.01)
        try tap("sensory-strong", in: app)
        try expectStatus("gain is zero", in: app)
        assertNoPlayingVoicesIfExposed(in: app)
        try reveal(volume, in: app, towardTop: true)
        volume.adjust(toNormalizedSliderPosition: 0.5)

        try restoreSwitch(sound, to: "0", in: app)
        XCTAssertEqual(sound.value as? String, "0")
        try restoreSwitch(haptics, to: "0", in: app)
        XCTAssertEqual(haptics.value as? String, "0")
        try reveal(element("sensory-diagnostics", in: app), in: app)
        let countersBefore = feedbackCountersIfExposed(in: app)
        try tap("sensory-weak", in: app, towardTop: true)
        try expectStatus("Haptics disabled", in: app)
        try tap("sensory-zero", in: app, towardTop: true)
        try expectStatus("no tone", in: app)
        try tap("sensory-sequence", in: app, towardTop: true)
        try tap("sensory-stop", in: app)
        try expectStatus("Stopped", in: app)
        assertNoPlayingVoicesIfExposed(in: app)
        if let countersBefore, let countersAfter = feedbackCountersIfExposed(in: app) {
            XCTAssertEqual(countersAfter.haptics, countersBefore.haptics)
            XCTAssertEqual(countersAfter.tones, countersBefore.tones)
        }
        try tap("sensory-done", in: app)
    }

    func testRepeatedSoundStartStopAcceptsFreshTouchesWithoutResumingStoppedSequences() throws {
        let app = XCUIApplication()
        // Read-only argument-domain overrides leave the user's saved preferences untouched.
        app.launchArguments = ["-machine.sound", "YES", "-machine.haptics", "YES", "-machine.soundGain", "0.5"]
        app.launch()
        defer { app.terminate(); app.launchArguments = [] }
        try openSensoryCheck(in: app)
        XCTAssertEqual(app.switches["sound-toggle"].value as? String, "1")
        XCTAssertEqual(app.switches["haptics-toggle"].value as? String, "1")

        let sequence = element("sensory-sequence", in: app)
        let stop = element("sensory-stop", in: app)
        for iteration in 1...3 {
            try XCTContext.runActivity(named: "Sound start/stop cycle \(iteration)") { _ in
                try reveal(sequence, in: app, towardTop: true)
                try reveal(stop, in: app)
                XCTAssertTrue(sequence.isEnabled)
                sequence.tap()
                // Submit Stop directly: a status wait would let the short sequence finish first.
                stop.tap()
                try expectStatus("Stopped", in: app)
                XCTAssertTrue(sequence.isEnabled)
                try expectStoppedFeedbackRemainsFrozen(in: app)

                let diagnostics = element("sensory-diagnostics", in: app)
                let before = try XCTUnwrap(diagnostics.value as? String)
                let acceptedBefore = try XCTUnwrap(diagnosticCount("tones", in: before))
                try tap("sensory-strong", in: app, towardTop: true)
                try expectStatus("Tone requested", in: app)
                try reveal(diagnostics, in: app)
                let acceptedFreshTouch = XCTNSPredicateExpectation(
                    predicate: NSPredicate(format: "value ENDSWITH %@", "tones=\(acceptedBefore + 1)"),
                    object: diagnostics
                )
                XCTAssertEqual(XCTWaiter.wait(for: [acceptedFreshTouch], timeout: 3), .completed)
                try tap("sensory-stop", in: app, towardTop: true)
                try expectStatus("Stopped", in: app)
                try expectStoppedFeedbackRemainsFrozen(in: app)
            }
        }
        try tap("sensory-done", in: app)
    }

    private func element(_ identifier: String, in app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func openSensoryCheck(in app: XCUIApplication) throws {
        sensorySheetExpected = false
        try tap("sensory-check", in: app)
        guard app.switches["sound-toggle"].waitForExistence(timeout: 5) else {
            XCTFail("The feedback sheet did not open")
            throw NSError(domain: "AttentionQualityUITests", code: 3)
        }
        XCTAssertTrue(app.switches["haptics-toggle"].exists)
        sensorySheetExpected = true
    }

    private func tap(_ identifier: String, in app: XCUIApplication, towardTop: Bool = false) throws {
        let target = element(identifier, in: app)
        try reveal(target, in: app, towardTop: towardTop)
        target.tap()
        if identifier == "sensory-done" { sensorySheetExpected = false }
    }

    private func tapSwitch(_ toggle: XCUIElement, in app: XCUIApplication) throws {
        try reveal(toggle, in: app, towardTop: true, requireVisibleThumb: true)
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
    }

    private func expectStatus(_ text: String, in app: XCUIApplication) throws {
        let target = element("sensory-status", in: app)
        try reveal(target, in: app)
        let status = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label CONTAINS %@", text), object: target)
        XCTAssertEqual(XCTWaiter.wait(for: [status], timeout: 3), .completed)
    }

    private func restoreSwitch(_ toggle: XCUIElement, to value: String, in app: XCUIApplication) throws {
        try reveal(toggle, in: app, towardTop: true, requireVisibleThumb: true)
        for _ in 0..<2 {
            if toggle.value as? String == value { break }
            try tapSwitch(toggle, in: app)
            let changed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", value), object: toggle)
            if XCTWaiter.wait(for: [changed], timeout: 2) == .completed { break }
        }
        XCTAssertEqual(toggle.value as? String, value)
    }

    private func expectStoppedFeedbackRemainsFrozen(in app: XCUIApplication) throws {
        let diagnostics = element("sensory-diagnostics", in: app)
        try reveal(diagnostics, in: app)
        let stopped = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "value CONTAINS %@", "playing=0;"), object: diagnostics
        )
        XCTAssertEqual(XCTWaiter.wait(for: [stopped], timeout: 3), .completed)
        let snapshot = try XCTUnwrap(diagnostics.value as? String)
        XCTAssertEqual(try XCTUnwrap(diagnosticCount("playing", in: snapshot)), 0)
        let resumedFeedback = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "value != %@", snapshot), object: diagnostics
        )
        resumedFeedback.isInverted = true
        // One second covers three remaining 320 ms note intervals.
        XCTAssertEqual(XCTWaiter.wait(for: [resumedFeedback], timeout: 1), .completed)
    }

    private func normalizedVolume(_ slider: XCUIElement) throws -> CGFloat {
        let text = try XCTUnwrap(slider.value as? String)
        let number = text.replacingOccurrences(of: ",", with: ".").filter { $0.isNumber || $0 == "." || $0 == "-" }
        var value = try XCTUnwrap(Double(number))
        if text.contains("%") { value /= 100 }
        return CGFloat(min(1, max(0, value)))
    }

    private func diagnosticsText(in app: XCUIApplication) -> String? {
        let diagnostics = element("sensory-diagnostics", in: app)
        guard diagnostics.exists else { return nil }
        return (diagnostics.value as? String) ?? diagnostics.label
    }

    private func diagnosticCount(_ key: String, in text: String) -> Int? {
        guard let expression = try? NSRegularExpression(pattern: "\\b\(key)=([0-9]+)"),
              let match = expression.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let range = Range(match.range(at: 1), in: text) else { return nil }
        return Int(text[range])
    }

    private func assertNoPlayingVoicesIfExposed(in app: XCUIApplication) {
        guard let text = diagnosticsText(in: app), let playing = diagnosticCount("playing", in: text) else { return }
        XCTAssertEqual(playing, 0)
    }

    private func feedbackCountersIfExposed(in app: XCUIApplication) -> (haptics: Int, tones: Int)? {
        guard let text = diagnosticsText(in: app), let haptics = diagnosticCount("haptics", in: text),
              let tones = diagnosticCount("tones", in: text) else { return nil }
        return (haptics, tones)
    }

    private func expectMatrix(rows: Int, columns: Int, in app: XCUIApplication) {
        let matrix = element("attention-matrix", in: app)
        XCTAssertTrue(matrix.waitForExistence(timeout: 10))
        let expected = "Attention matrix, \(rows) query rows and \(columns) key columns"
        let match = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label BEGINSWITH %@", expected), object: matrix)
        XCTAssertEqual(XCTWaiter.wait(for: [match], timeout: 5), .completed)
    }

    private func reveal(_ target: XCUIElement, in app: XCUIApplication, towardTop: Bool = false,
                        requireVisibleThumb: Bool = false) throws {
        // Lazy Form rows do not exist until scrolled into view, so page before requiring existence.
        for _ in 0..<8 {
            try recoverSensorySheetAfterHarnessScroll(in: app)
            let window = screenBounds(in: app)
            let done = element(sensorySheetExpected ? "sensory-done" : "done-machine-controls", in: app)
            let top = done.exists ? min(done.frame.maxY + 12, window.minY + 160) : window.minY + 60
            let bottom = window.maxY - 60
            if target.exists && target.isHittable {
                if !requireVisibleThumb || (target.frame.midY >= top && target.frame.midY <= bottom) { return }
            }
            if target.exists && !target.frame.isEmpty {
                if requireVisibleThumb && target.frame.midY < top { try pageForm(in: app, towardTop: true) }
                else if requireVisibleThumb && target.frame.midY > bottom { try pageForm(in: app, towardTop: false) }
                else { try pageForm(in: app, towardTop: target.frame.midY < window.midY) }
            } else { try pageForm(in: app, towardTop: towardTop) }
        }
        // Reading identifier on a missing query raises an Objective-C exception and bypasses defer.
        XCTFail("Control did not become hittable after bounded Form scrolling")
        throw NSError(domain: "AttentionQualityUITests", code: 1)
    }

    private func recoverSensorySheetAfterHarnessScroll(in app: XCUIApplication) throws {
        guard sensorySheetExpected else { return }
        if app.state != .runningForeground { app.activate() }
        guard !element("sensory-done", in: app).exists else { return }
        // Only scrolling helpers recover a dismissed sheet. The explicit Home test above stays strict.
        let opener = element("sensory-check", in: app)
        guard opener.exists && opener.isHittable else {
            XCTFail("Feedback sheet was lost and its opening control is unavailable")
            throw NSError(domain: "AttentionQualityUITests", code: 4)
        }
        opener.tap()
        guard element("sensory-done", in: app).waitForExistence(timeout: 3) else {
            XCTFail("Feedback sheet could not be reopened after a harness scroll")
            throw NSError(domain: "AttentionQualityUITests", code: 5)
        }
    }

    private func screenBounds(in app: XCUIApplication) -> CGRect {
        // iOS 27 beta can report the same window at one-third scale between AX snapshots.
        // Retain the observed portrait screen extent for confined, absolute-coordinate drags.
        for frame in [app.frame] + app.windows.allElementsBoundByIndex.map(\.frame) {
            if frame.width * frame.height > largestObservedScreen.width * largestObservedScreen.height {
                largestObservedScreen = frame
            }
        }
        return largestObservedScreen
    }

    private func pageForm(in app: XCUIApplication, towardTop: Bool) throws {
        let window = screenBounds(in: app)
        // The underlying screen can expose a scaled navigation bar behind a sheet.
        let done = element(sensorySheetExpected ? "sensory-done" : "done-machine-controls", in: app)
        let headerBottom = done.exists ? min(done.frame.maxY + 24, window.minY + 160) : window.minY + 100
        let top = max(window.minY + 24, headerBottom)
        let bottom = window.maxY - 100
        guard bottom - top > 80 else {
            XCTFail("No safe Form viewport: screen=\(window), header=\(headerBottom)")
            throw NSError(domain: "AttentionQualityUITests", code: 6)
        }
        let middle = (top + bottom) / 2
        let halfDistance = min(80, (bottom - top) / 4)
        let startY = middle + (towardTop ? -halfDistance : halfDistance)
        let endY = middle + (towardTop ? halfDistance : -halfDistance)
        let x = window.minX + 24
        let origin = app.coordinate(withNormalizedOffset: .zero)
        let start = origin.withOffset(CGVector(dx: x - window.minX, dy: startY - window.minY))
        let end = origin.withOffset(CGVector(dx: x - window.minX, dy: endY - window.minY))
        start.press(forDuration: 0.05, thenDragTo: end)
    }
}
