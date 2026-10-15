import XCTest

/// Exercises cancellation and publication of the largest portable teaching calculation.
@MainActor
final class AttentionComputationUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testMaximumShapePublishesLatestControlsAfterRapidChangesAndRestore() throws {
        let app = XCUIApplication()
        // Argument-domain fixtures do not replace the user's saved sentences or sensory preferences.
        app.launchArguments = [
            "-machine.source", Array(repeating: "light", count: 24).joined(separator: " "),
            "-machine.target", Array(repeating: "we", count: 23).joined(separator: " "),
            "-machine.sound", "NO", "-machine.haptics", "NO"
        ]
        app.launch()
        defer { app.terminate(); app.launchArguments = [] }

        expectReady(heads: 2, layers: 2, in: app)
        let matrix = element("attention-matrix", in: app)
        XCTAssertTrue(matrix.label.hasPrefix("Attention matrix, 24 query rows and 24 key columns"))
        app.buttons["Previous machine stage"].tap()
        XCTAssertEqual(element("machine-stage-title", in: app).label, "Embedding + position")

        element("machine-controls", in: app).tap()
        let heads = element("teaching-heads", in: app)
        try reveal(heads, in: app, towardTop: true)
        heads.buttons["8"].tap()
        let layers = element("teaching-layers", in: app)
        try reveal(layers, in: app)
        for _ in 0..<4 { try increment(layers, in: app) }
        XCTAssertTrue(layers.label.contains("6 encoder + decoder layers"))

        let position = app.sliders["Position strength"]
        try reveal(position, in: app)
        // Every change creates a competing request. The last slider value must win.
        for value: CGFloat in [0.9, 0.1, 1, 0] {
            position.adjust(toNormalizedSliderPosition: value)
        }
        // Native sliders may land near their endpoint; compare the actual exposed value.
        let finalPositionStrength = try actualPositionStrength(position)
        element("done-machine-controls", in: app).tap()
        expectReady(heads: 8, layers: 6, in: app)
        XCTAssertEqual(try publishedPosition(in: app), finalPositionStrength, accuracy: 0.03)
        let finalEquation = element("machine-equation", in: app).label
        XCTAssertTrue(finalEquation.contains("√8 · E(token)"))
        XCTAssertTrue(finalEquation.contains(" · PE(position)"))

        element("machine-controls", in: app).tap()
        try reveal(position, in: app)
        XCTAssertEqual(try actualPositionStrength(position), finalPositionStrength, accuracy: 0.02)
        position.adjust(toNormalizedSliderPosition: 1)
        position.adjust(toNormalizedSliderPosition: 0.2)
        let restore = app.buttons["Restore teaching model"]
        try reveal(restore, in: app)
        restore.tap()
        element("done-machine-controls", in: app).tap()

        expectReady(heads: 2, layers: 2, in: app)
        XCTAssertEqual(try publishedPosition(in: app), 1, accuracy: 0.000001)
        let equation = element("machine-equation", in: app)
        XCTAssertEqual(equation.label, "x = √dₘₒdₑₗ · E(token) + PE(position)")
        // An older maximum-shape worker may finish later, but it must never replace restored state.
        let stalePublication = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "label == %@", "Calculation ready · 8 heads · 6 layers"),
            object: element("machine-calculation-state", in: app)
        )
        stalePublication.isInverted = true
        XCTAssertEqual(XCTWaiter.wait(for: [stalePublication], timeout: 0.75), .completed)

        app.tabBars.buttons["Paper"].tap()
        app.tabBars.buttons["Machine"].tap()
        expectReady(heads: 2, layers: 2, in: app)
        XCTAssertEqual(element("play-machine", in: app).label, "Play machine")
        element("machine-controls", in: app).tap()
        try reveal(heads, in: app, towardTop: true)
        // SwiftUI segments expose an empty AX value. Fresh published trace verifies the selection.
        XCTAssertTrue(heads.buttons["2"].isHittable)
        XCTAssertTrue(layers.label.contains("2 encoder + decoder layers"))
        try reveal(position, in: app)
        XCTAssertEqual(try actualPositionStrength(position), 1, accuracy: 0.03)
        element("done-machine-controls", in: app).tap()
        expectReady(heads: 2, layers: 2, in: app)
        XCTAssertEqual(try publishedPosition(in: app), 1, accuracy: 0.000001)
    }

    private func element(_ identifier: String, in app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func expectReady(heads: Int, layers: Int, in app: XCUIApplication) {
        let state = element("machine-calculation-state", in: app)
        XCTAssertTrue(state.waitForExistence(timeout: 10))
        let ready = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "label == %@", "Calculation ready · \(heads) heads · \(layers) layers"),
            object: state
        )
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 10), .completed)
    }

    private func publishedPosition(in app: XCUIApplication) throws -> Double {
        let value = try XCTUnwrap(element("machine-calculation-state", in: app).value as? String)
        XCTAssertTrue(value.hasPrefix("Position strength:"))
        let number = value.replacingOccurrences(of: ",", with: ".").filter { $0.isNumber || $0 == "." || $0 == "-" }
        return try XCTUnwrap(Double(number))
    }

    private func increment(_ stepper: XCUIElement, in app: XCUIApplication) throws {
        try reveal(stepper, in: app)
        let button = stepper.buttons["Increment"]
        if button.exists && button.isHittable { button.tap() }
        else {
            // Native Stepper's plus control remains at the trailing edge if its child label is omitted.
            stepper.coordinate(withNormalizedOffset: CGVector(dx: 1, dy: 0.5))
                .withOffset(CGVector(dx: -22, dy: 0)).tap()
        }
    }

    private func actualPositionStrength(_ slider: XCUIElement) throws -> Double {
        let text = try XCTUnwrap(slider.value as? String)
        let number = text.replacingOccurrences(of: ",", with: ".").filter { $0.isNumber || $0 == "." || $0 == "-" }
        var value = try XCTUnwrap(Double(number))
        // Percentage accessibility values are normalized; numeric values use the native 0...2 range.
        if text.contains("%") { value = value / 100 * 2 }
        return value
    }

    private func reveal(_ target: XCUIElement, in app: XCUIApplication, towardTop: Bool = false) throws {
        // Lazy Form rows appear only after scrolling; existence cannot be a precondition.
        for _ in 0..<8 {
            if target.exists && target.isHittable { return }
            if target.exists && !target.frame.isEmpty {
                if target.frame.midY < app.windows.firstMatch.frame.midY { app.swipeDown() }
                else { app.swipeUp() }
            } else if towardTop { app.swipeDown() }
            else { app.swipeUp() }
        }
        XCTFail("Control did not become hittable: \(target.identifier)")
        throw NSError(domain: "AttentionComputationUITests", code: 1)
    }
}
