// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AttentionCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "AttentionCore", targets: ["AttentionCore"])],
    targets: [
        .target(name: "AttentionCore"),
        .testTarget(name: "AttentionCoreTests", dependencies: ["AttentionCore"])
    ]
)
