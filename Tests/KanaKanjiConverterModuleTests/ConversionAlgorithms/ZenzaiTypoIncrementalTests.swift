@testable import KanaKanjiConverterModule
import XCTest

final class ZenzaiTypoIncrementalTests: XCTestCase {
    private func assertEquivalentCandidateRanking(
        _ incremental: [ZenzaiTypoCandidate],
        _ full: [ZenzaiTypoCandidate],
        message: @autoclosure () -> String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let message = message()
        XCTAssertEqual(
            incremental.map(\.score),
            full.map(\.score),
            "\(message): score ranking differs",
            file: file,
            line: line
        )
        XCTAssertEqual(
            Dictionary(grouping: incremental, by: \.self).mapValues(\.count),
            Dictionary(grouping: full, by: \.self).mapValues(\.count),
            "\(message): candidate contents differ",
            file: file,
            line: line
        )
    }

    private struct DeterministicCharacterContext: ZenzCompatibleInputLanguageModelContext {
        init() {
            var characters: [Character] = []
            var seen: Set<Character> = []
            for character in "?abcdefghijklmnopqrstuvwxyzアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンッャュョー；"
                where seen.insert(character).inserted {
                characters.append(character)
            }
            self.characters = characters
            self.tokenIDs = Dictionary(uniqueKeysWithValues: characters.enumerated().map { ($0.element, $0.offset) })
        }

        let characters: [Character]
        let tokenIDs: [Character: Int]

        var vocabSize: Int { self.characters.count }

        func encodeRaw(_ text: String) -> [Int] {
            text.map { self.tokenIDs[$0, default: 0] }
        }

        func tokenToSingleCharacter(tokenID: Int) -> Character? {
            self.characters.indices.contains(tokenID) ? self.characters[tokenID] : nil
        }

        func nextLogProbs(promptTokenIDs: [Int], emittedTokenIDs: [Int]) -> [Float]? {
            let previous = emittedTokenIDs.last ?? promptTokenIDs.last ?? 0
            return self.characters.indices.map { tokenID in
                -Float((tokenID * 17 + previous * 7 + emittedTokenIDs.count * 13) % 101) / 10
            }
        }
    }

    func testIncrementalCheckpointMatchesFullSearchExactly() {
        let scenarios: [(String, InputStyle)] = [
            ("かなかんじへんかん", .direct),
            ("konobunshou", .roman2kana),
            ("konobjxp", .mapped(id: .defaultAZIK))
        ]
        let context = DeterministicCharacterContext()
        let config = ExperimentalTypoCorrectionConfig(
            languageModel: .zenz,
            beamSize: 8,
            topK: 16,
            nBest: 5
        )

        for (input, inputStyle) in scenarios {
            let incrementalCache = ZenzaiTypoGenerationCache()
            let fullCache = ZenzaiTypoGenerationCache()
            var fullConfig = config
            fullConfig.usesIncrementalCheckpoint = false
            var composingText = ComposingText()
            for character in input {
                composingText.insertAtCursorPosition(String(character), inputStyle: inputStyle)
                let incremental = ZenzaiTypoCandidateGenerator.generate(
                    context: context,
                    leftSideContext: "一致検証",
                    composingText: composingText,
                    inputStyle: inputStyle,
                    experimentalConfig: config,
                    cache: incrementalCache
                )
                let full = ZenzaiTypoCandidateGenerator.generate(
                    context: context,
                    leftSideContext: "一致検証",
                    composingText: composingText,
                    inputStyle: inputStyle,
                    experimentalConfig: fullConfig,
                    cache: fullCache
                )
                // Candidate generation ranks only by score. Candidates with exactly equal scores
                // may therefore exchange positions when Dictionary's randomized iteration order
                // differs between the incremental and full-search caches. Preserve the meaningful
                // contract here: identical score ranks and identical candidate contents, including
                // multiplicity, while allowing permutations only within equal-score ranks.
                self.assertEquivalentCandidateRanking(
                    incremental,
                    full,
                    message: "incremental search diverged for style=\(inputStyle), input=\(String(input.prefix(composingText.input.count)))"
                )
            }
        }
    }

    /// 打ち足した数が 1 文字でないときも，別の入力・別の左文脈と交互に探しても，checkpoint から再開した結果が全体の探索と同じになる
    func testCheckpointsResumeAcrossPrefixesInputsAndPromptsExactly() {
        let context = DeterministicCharacterContext()
        let config = ExperimentalTypoCorrectionConfig(languageModel: .zenz, beamSize: 8, topK: 16, nBest: 5)
        var fullConfig = config
        fullConfig.usesIncrementalCheckpoint = false
        let cache = ZenzaiTypoGenerationCache()
        // (左文脈, 入力)．同じ入力の続きを飛び飛びに，ほかの入力と左文脈をはさんで探す
        let requests: [(String, String)] = [
            ("一致検証", "konobu"),
            ("", "konobu"),
            ("一致検証", "kyouhaii"),
            ("一致検証", "konobunsh"),
            ("", "konobunshou"),
            ("一致検証", "konobunshou"),
            ("一致検証", "konobunshou"),
            ("一致検証", "kyouhaiitenki"),
            ("別の文脈", "konobunshouwo"),
        ]
        for (leftContext, input) in requests {
            var composingText = ComposingText()
            composingText.insertAtCursorPosition(input, inputStyle: .roman2kana)
            let resumed = ZenzaiTypoCandidateGenerator.generate(
                context: context, leftSideContext: leftContext, composingText: composingText,
                inputStyle: .roman2kana, experimentalConfig: config, cache: cache
            )
            let full = ZenzaiTypoCandidateGenerator.generate(
                context: context, leftSideContext: leftContext, composingText: composingText,
                inputStyle: .roman2kana, experimentalConfig: fullConfig, cache: ZenzaiTypoGenerationCache()
            )
            self.assertEquivalentCandidateRanking(resumed, full, message: "resumed search diverged for context=\(leftContext), input=\(input)")
        }
    }

    /// 前に探した入力の続きを探すとき，checkpoint から再開して LM の評価を減らす
    func testCheckpointIsReusedForLongerInput() {
        let context = DeterministicCharacterContext()
        var config = ExperimentalTypoCorrectionConfig(languageModel: .zenz, beamSize: 8, topK: 16, nBest: 5)
        config.collectsPerformanceMetrics = true
        func steps(_ input: String, cache: ZenzaiTypoGenerationCache) -> Int {
            var composingText = ComposingText()
            composingText.insertAtCursorPosition(input, inputStyle: .roman2kana)
            _ = ZenzaiTypoCandidateGenerator.generate(
                context: context, leftSideContext: "一致検証", composingText: composingText,
                inputStyle: .roman2kana, experimentalConfig: config, cache: cache
            )
            return cache.lastMetrics.stepCount
        }
        let warmed = ZenzaiTypoGenerationCache()
        _ = steps("konobunsh", cache: warmed)
        _ = steps("", cache: warmed)
        XCTAssertLessThan(steps("konobunshouwo", cache: warmed), steps("konobunshouwo", cache: ZenzaiTypoGenerationCache()))
    }

    /// 途中で取り消した探索の checkpoint から再開しても，全体の探索と同じ結果になる
    func testStoppedSearchResumesExactly() {
        let context = DeterministicCharacterContext()
        let config = ExperimentalTypoCorrectionConfig(languageModel: .zenz, beamSize: 8, topK: 16, nBest: 5)
        var fullConfig = config
        fullConfig.usesIncrementalCheckpoint = false
        var composingText = ComposingText()
        composingText.insertAtCursorPosition("konobunshouwo", inputStyle: .roman2kana)
        for stopAfter in [1, 3, 6, 10] {
            let cache = ZenzaiTypoGenerationCache()
            var calls = 0
            let stopped = ZenzaiTypoCandidateGenerator.generate(
                context: context, leftSideContext: "一致検証", composingText: composingText,
                inputStyle: .roman2kana, experimentalConfig: config, cache: cache,
                shouldStop: { calls += 1; return calls > stopAfter }
            )
            XCTAssertTrue(stopped.isEmpty)
            let resumed = ZenzaiTypoCandidateGenerator.generate(
                context: context, leftSideContext: "一致検証", composingText: composingText,
                inputStyle: .roman2kana, experimentalConfig: config, cache: cache
            )
            let full = ZenzaiTypoCandidateGenerator.generate(
                context: context, leftSideContext: "一致検証", composingText: composingText,
                inputStyle: .roman2kana, experimentalConfig: fullConfig, cache: ZenzaiTypoGenerationCache()
            )
            self.assertEquivalentCandidateRanking(resumed, full, message: "resumed after stop \(stopAfter)")
        }
    }
}
