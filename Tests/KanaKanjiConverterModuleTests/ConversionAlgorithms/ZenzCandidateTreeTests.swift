import Foundation
@testable import KanaKanjiConverterModule
import XCTest

final class ZenzCandidateTreeTests: XCTestCase {
    func testSharedPrefixUsesOneNode() {
        let tree = ZenzCandidateTree(rootToken: 9, candidates: [[1, 2, 3], [1, 2, 4], [5]], maximumNodes: 100, maximumCandidates: 10)
        // 根 + 1 + 2 + 3 + 4 + 5
        XCTAssertEqual(tree.nodes.count, 6)
        XCTAssertEqual(tree.acceptedCount, 3)
        XCTAssertEqual(tree.nodes[0].candidates, [0, 1, 2])
        // 共通の先頭(1, 2)は両方の候補を持つ
        XCTAssertEqual(tree.paths[0], [1, 2, 3])
        XCTAssertEqual(tree.paths[1], [1, 2, 4])
        XCTAssertEqual(tree.nodes[1].candidates, [0, 1])
        XCTAssertEqual(tree.nodes[2].candidates, [0, 1])
        XCTAssertEqual(tree.nodes[3].candidates, [0])
        XCTAssertEqual(tree.nodes[4].candidates, [1])
        XCTAssertEqual(tree.nodes[4].depth, 3)
        XCTAssertEqual(tree.nodes[4].parent, 2)
    }

    func testParentsComeBeforeChildren() {
        let tree = ZenzCandidateTree(rootToken: 0, candidates: [[1, 2], [3, 4, 5], [1, 6, 7], [3, 8]], maximumNodes: 100, maximumCandidates: 10)
        for (index, node) in tree.nodes.enumerated() {
            if let parent = node.parent {
                XCTAssertLessThan(parent, index)
                XCTAssertEqual(tree.nodes[parent].depth + 1, node.depth)
            }
        }
    }

    func testCandidatesBeyondTheBudgetAreLeftOut() {
        // 根 + [1, 2, 3] で 4 節．次の [4, 5] は 2 節足りないので入れず，[1, 2, 6] は 1 節で入る
        let tree = ZenzCandidateTree(rootToken: 0, candidates: [[1, 2, 3], [4, 5], [1, 2, 6]], maximumNodes: 5, maximumCandidates: 10)
        XCTAssertEqual(tree.nodes.count, 5)
        XCTAssertNotNil(tree.paths[0])
        XCTAssertNil(tree.paths[1])
        XCTAssertNotNil(tree.paths[2])
        XCTAssertEqual(tree.acceptedIndices, [0, nil, 1])
        // 入れなかった候補は節に名前を残さない
        XCTAssertFalse(tree.nodes.contains { $0.candidates.contains(2) })
    }

    func testCandidateCountLimit() {
        let tree = ZenzCandidateTree(rootToken: 0, candidates: [[1], [2], [3]], maximumNodes: 100, maximumCandidates: 2)
        XCTAssertEqual(tree.acceptedCount, 2)
        XCTAssertNil(tree.paths[2])
    }

    func testScoresAddTokenAndEndProbabilities() {
        // 候補 [1, 2] と [1] と [](空)．空の候補は根で終わる
        let tree = ZenzCandidateTree(rootToken: 0, candidates: [[1, 2], [1], []], maximumNodes: 100, maximumCandidates: 10)
        let end: Int32 = 99
        // 節 n でトークン t の対数確率を -(n * 10 + t) とする(t が end なら -(n * 10 + 0.5))
        let scores = tree.scores(endToken: end) { node, token in
            token == end ? -(Float(node) * 10 + 0.5) : -(Float(node) * 10 + Float(token))
        }
        // [1, 2]: 根で 1(-1)，節 1 で 2(-12)，節 2 で終わり(-20.5)
        XCTAssertEqual(scores[0]!, -1 - 12 - 20.5, accuracy: 1e-5)
        // [1]: 根で 1(-1)，節 1 で終わり(-10.5)
        XCTAssertEqual(scores[1]!, -1 - 10.5, accuracy: 1e-5)
        // []: 根で終わり(-0.5)
        XCTAssertEqual(scores[2]!, -0.5, accuracy: 1e-5)
    }

    func testLogSumExpMatchesDirectComputation() {
        var generator = SystemRandomNumberGenerator()
        for count in [1, 7, 8, 9, 6000] {
            let values = (0 ..< count).map { _ in Float.random(in: -30 ... 20, using: &generator) }
            let maximum = values.max()!
            let expected = maximum + log(values.reduce(Float(0)) { $0 + exp($1 - maximum) })
            let actual = values.withUnsafeBufferPointer { ZenzLogSumExp.compute($0.baseAddress!, count: count) }
            XCTAssertEqual(actual, expected, accuracy: 1e-4, "count: \(count)")
        }
    }

    func testApproximateExpIsAccurate() {
        let inputs: [Float] = [0, -0.1, -0.5, -1, -2.5, -10, -30, -60, -86]
        for x in inputs {
            let approximated = ZenzLogSumExp.exp(SIMD8<Float>(repeating: x))[0]
            XCTAssertEqual(approximated / exp(x), 1, accuracy: 1e-5, "x: \(x)")
        }
        // 下限より小さい値は 0 に近い値になる(発散しない)
        XCTAssertLessThan(ZenzLogSumExp.exp(SIMD8<Float>(repeating: -1000))[0], 1e-37)
    }
}
