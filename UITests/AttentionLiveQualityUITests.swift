import XCTest

/// One bounded live-model scenario; prompts are public fixtures and never persisted.
@MainActor
final class AttentionLiveQualityUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testShortLivePromptKeepsItsSelectionAcrossTabsAndBackgroundUnload() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("Live MLX inference requires a physical iPhone.")
        #else
        let app = XCUIApplication()
        app.launchArguments = ["--verify-local-model"]
        app.launch()
        defer { app.terminate(); app.launchArguments = [] }

        let mode = element("model-trace-mode", in: app)
        let inspect = element("inspect-model", in: app)
        let measured = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "label == %@", "MEASURED ON THIS IPHONE"), object: mode
        )
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == true"), object: inspect)
        XCTAssertEqual(XCTWaiter.wait(for: [measured, ready], timeout: 50), .completed)
        XCTAssertFalse(element("model-error", in: app).exists)

        let prompt = element("model-prompt", in: app)
        try reveal(prompt, in: app)
        replaceText(prompt, with: "Hello.")
        XCTAssertEqual(prompt.value as? String, "Hello.")
        try reveal(inspect, in: app)
        inspect.tap() // The native action is limited to 32 continuation tokens.

        let shortPrompt = app.staticTexts["Hello."].firstMatch
        let newCapture = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == true"), object: shortPrompt)
        let readyAgain = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == true"), object: inspect)
        XCTAssertEqual(XCTWaiter.wait(for: [newCapture, readyAgain], timeout: 50), .completed)
        XCTAssertEqual(mode.label, "MEASURED ON THIS IPHONE")
        XCTAssertFalse(element("model-error", in: app).exists)

        let matrix = element("attention-matrix", in: app)
        let tokenCount = try queryRows(matrix.label)
        XCTAssertGreaterThan(tokenCount, 0)
        XCTAssertLessThan(tokenCount, 24, "The short fixture must differ from the longer launch fixture.")
        let query = element("model-query-readout", in: app)
        XCTAssertTrue(query.waitForExistence(timeout: 5))
        XCTAssertEqual(Int(query.label.filter(\.isNumber)), tokenCount, "Select the final query in the live capture.")
        let secondToken = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Token 2,")).firstMatch
        try reveal(secondToken, in: app)
        secondToken.tap()
        XCTAssertEqual(Int(query.label.filter(\.isNumber)), 2, "Keep the user-selected query rather than restoring the last row.")
        let selectedQuery = query.label

        app.tabBars.buttons["Paper"].tap()
        app.tabBars.buttons["Local model"].tap()
        XCTAssertEqual(mode.label, "MEASURED ON THIS IPHONE")
        XCTAssertTrue(shortPrompt.exists)
        XCTAssertEqual(query.label, selectedQuery)
        XCTAssertTrue(inspect.exists)

        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertTrue(element("download-model", in: app).waitForExistence(timeout: 15))
        XCTAssertFalse(inspect.exists)
        XCTAssertEqual(mode.label, "MEASURED ON THIS IPHONE")
        XCTAssertTrue(shortPrompt.exists)
        XCTAssertEqual(query.label, selectedQuery)
        XCTAssertFalse(element("model-error", in: app).exists)
        #endif
    }

    private func element(_ identifier: String, in app: XCUIApplication) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func queryRows(_ label: String) throws -> Int {
        let expression = try NSRegularExpression(pattern: "Attention matrix, ([0-9]+) query rows")
        let text = label as NSString
        let match = try XCTUnwrap(expression.firstMatch(in: label, range: NSRange(location: 0, length: text.length)))
        return try XCTUnwrap(Int(text.substring(with: match.range(at: 1))))
    }

    private func replaceText(_ field: XCUIElement, with text: String) {
        let existing = field.value as? String ?? ""
        field.tap()
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.95, dy: 0.90)).tap()
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: existing.count))
        field.typeText(text)
    }

    private func reveal(_ target: XCUIElement, in app: XCUIApplication) throws {
        XCTAssertTrue(target.waitForExistence(timeout: 5))
        for _ in 0..<6 {
            if target.isHittable { return }
            if target.frame.midY < app.windows.firstMatch.frame.midY { app.swipeDown() }
            else { app.swipeUp() }
        }
        XCTFail("Control did not become hittable: \(target.identifier)")
        throw NSError(domain: "AttentionLiveQualityUITests", code: 1)
    }
}
