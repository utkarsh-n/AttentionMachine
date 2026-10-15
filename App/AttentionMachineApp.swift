import SwiftUI

@main
struct AttentionMachineApp: App {
    var body: some Scene {
        WindowGroup {
            RootView().preferredColorScheme(.dark).tint(Theme.mint)
        }
    }
}

struct RootView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var tab = ProcessInfo.processInfo.arguments.contains("--verify-local-model") || ProcessInfo.processInfo.arguments.contains("--verify-stability") || ProcessInfo.processInfo.arguments.contains("--verify-long-stability") ? 2 : 0
    @State private var localIntelligence = AttentionLocalIntelligence()
    private var localModel: LocalModelEngine { localIntelligence.inspection }
    var body: some View {
        TabView(selection: $tab) {
            Tab("Machine", systemImage: "cpu", value: 0) { MachineView() }
            Tab("Paper", systemImage: "doc.text", value: 1) { PaperView() }
            Tab("Local model", systemImage: "sparkles", value: 2) { LocalIntelligenceView(intelligence: localIntelligence) }
        }.background(Theme.bg)
            .task {
                if ProcessInfo.processInfo.arguments.contains("--verify-long-stability") {
                    await DeviceStabilityCheck.runLong(engine: localModel)
                } else if ProcessInfo.processInfo.arguments.contains("--verify-stability") {
                    await DeviceStabilityCheck.run(engine: localModel)
                }
            }
            .onChange(of: scenePhase) { _, phase in
                switch phase {
                case .inactive:
                    localIntelligence.cancel()
                case .background:
                    Task { await localIntelligence.unload() }
                case .active:
                    break
                @unknown default:
                    break
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                Task { await localIntelligence.unload() }
            }
    }
}
