import XCTest

/// Focused interaction checks. The physical model check uses only the bundled offline weights.
@MainActor
final class AttentionMachineUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testAdvancingTheMachineChangesTheComputationStage() throws {
        let app = XCUIApplication()
        app.launch()
        let title = element("machine-stage-title", in: app)
        XCTAssertTrue(title.waitForExistence(timeout: 10))
        XCTAssertEqual(title.label, "Multi-head attention")
        try tap("next-stage", in: app)
        XCTAssertEqual(title.label, "Add & normalize")
        try tap("next-stage", in: app)
        XCTAssertEqual(title.label, "Feed-forward network")
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "32 CHANNELS")).firstMatch.exists)
    }

    func testCausalGateChangesTheDecoderExplanationAndCanBeRestored() throws {
        let app = XCUIApplication()
        app.launch()
        for _ in 0..<3 { try tap("next-stage", in: app) }
        XCTAssertEqual(element("machine-stage-title", in: app).label, "Masked decoder")
        try tap("machine-controls", in: app)
        let mask = app.switches["causal-mask"]
        try reveal(mask, in: app)
        XCTAssertEqual(mask.value as? String, "1")
        tapSwitchControl(mask)
        XCTAssertEqual(mask.value as? String, "0")
        try tap("done-machine-controls", in: app)
        XCTAssertTrue(element("machine-equation", in: app).label.contains("Mask disabled"))
        try tap("machine-controls", in: app)
        try reveal(mask, in: app)
        tapSwitchControl(mask)
        try tap("done-machine-controls", in: app)
        XCTAssertEqual(element("machine-equation", in: app).label, "Attention sees only the past + present")
    }

    func testSourceEditSurvivesRelaunch() throws {
        let app = XCUIApplication()
        app.launch()
        try tap("edit-input", in: app)
        let source = element("source-input", in: app)
        XCTAssertTrue(source.waitForExistence(timeout: 5))
        let original = source.value as? String ?? ""
        let replacement = "attention connects light"
        replaceText(source, with: replacement)
        try tap("apply-input", in: app)
        XCTAssertTrue(app.buttons["Token 1, attention"].waitForExistence(timeout: 5))

        app.terminate()
        app.launch()
        XCTAssertTrue(app.buttons["Token 1, attention"].waitForExistence(timeout: 10))
        try tap("edit-input", in: app)
        let restoredSource = element("source-input", in: app)
        XCTAssertEqual(restoredSource.value as? String, replacement)
        // Restore the existing source after checking persistence.
        replaceText(restoredSource, with: original)
        try tap("apply-input", in: app)
    }

    func testPaperNavigationAndSequenceSliderUpdateCalculatedCost() throws {
        let app = XCUIApplication()
        app.launch()
        app.tabBars.buttons["Paper"].tap()
        try tap("paper-chapter-Abstract", in: app)
        XCTAssertEqual(element("paper-detail-title", in: app).label, "Compare the engines")
        let ratio = element("paper-readout-Attention / recurrent cost", in: app)
        let initialRatio = ratio.label
        let length = app.sliders["paper-slider-Sequence length n"]
        try reveal(length, in: app)
        length.adjust(toNormalizedSliderPosition: 0.75)
        XCTAssertNotEqual(ratio.label, initialRatio)
        let serial = element("paper-readout-Recurrent serial steps", in: app)
        XCTAssertGreaterThan(Int(serial.label) ?? 0, 128)
    }

    func testBundledModelCaptureIsExplicitlyRecordedUntilLiveInferenceRuns() throws {
        let app = XCUIApplication()
        app.launch()
        app.tabBars.buttons["Local model"].tap()
        let mode = element("model-trace-mode", in: app)
        XCTAssertTrue(mode.waitForExistence(timeout: 10))
        XCTAssertEqual(mode.label, "RECORDED REAL MODEL TRACE")
        XCTAssertFalse(app.staticTexts["MEASURED ON THIS IPHONE"].exists)
        let note = element("model-trace-note", in: app)
        XCTAssertTrue(note.exists)
        XCTAssertTrue(note.label.contains("capture is fixed"))
        XCTAssertTrue(element("download-model", in: app).exists)
        XCTAssertFalse(element("inspect-model", in: app).exists)
    }

    func testCrossAttentionKeepsEveryKeyVisibleWhenLengthsDiffer() throws {
        let app = XCUIApplication()
        app.launch()
        for _ in 0..<4 { try tap("next-stage", in: app) }
        XCTAssertEqual(element("machine-stage-title", in: app).label, "Encoder → decoder")
        let matrix = element("attention-matrix", in: app)
        XCTAssertTrue(matrix.waitForExistence(timeout: 5))
        let pattern = try NSRegularExpression(pattern: "Attention matrix, ([0-9]+) query rows and ([0-9]+) key columns")
        let label = matrix.label as NSString
        let match = try XCTUnwrap(pattern.firstMatch(in: matrix.label, range: NSRange(location: 0, length: label.length)))
        let rows = try XCTUnwrap(Double(label.substring(with: match.range(at: 1))))
        let columns = try XCTUnwrap(Double(label.substring(with: match.range(at: 2))))
        XCTAssertGreaterThan(rows, 0)
        XCTAssertGreaterThan(columns, 0)
        // The rendered grid must fit every key column while keeping cells square.
        XCTAssertEqual(Double(matrix.frame.width / matrix.frame.height), columns / rows, accuracy: 0.08)
    }

    func testSwipingOverAttentionCanRevealTheMachineExplanation() throws {
        let app = XCUIApplication()
        app.launch()
        try reveal(element("machine-device-note", in: app), in: app)
    }

    func testLiveModelCapturesThenUnloadsWhenBackgrounded() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Live MLX inference requires a physical iPhone.")
        #else
        let app = XCUIApplication()
        app.launchArguments = ["--verify-local-model"]
        app.launch()
        let mode = element("model-trace-mode", in: app)
        let measured = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == %@", "MEASURED ON THIS IPHONE"), object: mode)
        XCTAssertEqual(XCTWaiter.wait(for: [measured], timeout: 50), .completed)
        XCTAssertTrue(element("inspect-model", in: app).exists)
        XCTAssertFalse(element("model-error", in: app).exists)
        app.tabBars.buttons["Paper"].tap()
        app.tabBars.buttons["Local model"].tap()
        XCTAssertEqual(mode.label, "MEASURED ON THIS IPHONE")
        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertTrue(element("download-model", in: app).waitForExistence(timeout: 10))
        XCTAssertFalse(element("inspect-model", in: app).exists)
        XCTAssertEqual(mode.label, "MEASURED ON THIS IPHONE")
        #endif
    }

    private func element(_ identifier: String, in app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func tap(_ identifier: String, in app: XCUIApplication) throws {
        let target = element(identifier, in: app)
        try reveal(target, in: app)
        target.tap()
    }

    private func reveal(_ target: XCUIElement, in app: XCUIApplication) throws {
        XCTAssertTrue(target.waitForExistence(timeout: 10))
        for _ in 0..<8 {
            if target.isHittable { return }
            app.swipeUp()
        }
        XCTFail("Element did not become hittable: \(target.identifier)")
        throw NSError(domain: "AttentionMachineUITests", code: 1)
    }

    private func replaceText(_ field: XCUIElement, with text: String) {
        let existing = field.value as? String ?? ""
        field.tap()
        // The lower right puts the insertion point after the final line of a multiline field.
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.90)).tap()
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: existing.count))
        field.typeText(text)
    }

    private func tapSwitchControl(_ toggle: XCUIElement) {
        // SwiftUI exposes the whole Form row as a switch; the native thumb is at its trailing edge.
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap()
    }
}
