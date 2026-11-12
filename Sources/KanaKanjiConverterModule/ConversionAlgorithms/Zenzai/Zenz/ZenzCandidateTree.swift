#if Zenzai || ZenzaiCPU
import llama
#endif

import Foundation

/// `KanaKanjiConverter.experimentalScoreCandidates` の結果
public struct ExperimentalCandidateScores: Sendable, Equatable {
    /// 表記ごとの log P(表記 | 読み, 文脈)。採点できなかった表記は nil
    public var scores: [Float?]
    /// 1 回の推論で評価した木のトークンの数(検証木の節の数)。時間の見積もりに使う
    public var evaluatedTokens: Int
    /// KV に無かったので，木の前に評価し直したプロンプトのトークンの数。時間の見積もりに使う
    public var decodedPromptTokens: Int

    public init(scores: [Float?], evaluatedTokens: Int, decodedPromptTokens: Int) {
        self.scores = scores
        self.evaluatedTokens = evaluatedTokens
        self.decodedPromptTokens = decodedPromptTokens
    }
}

/// 複数の候補のトークン列を，共通の先頭を 1 つの節にまとめた木(検証木)にする。
///
/// 木の節は `llama_decode` の 1 回のバッチに並べる順に持つ。親は必ず子より前に来る。
/// 節 0 は根で，プロンプトの最後のトークンを置く。根の出力(ロジット)で各候補の最初のトークンを測る。
/// 各節には，その節を通る候補の番号を持たせる。候補ごとに別の系列 ID を割り当て，節にはそれらをすべて付けると，
/// llama.cpp の注意のマスクは「同じ系列 ID を持ち，位置が自分以下のセル」だけを見せるので，各節は自分の祖先だけを見る。
struct ZenzCandidateTree: Equatable, Sendable {
    struct Node: Equatable, Sendable {
        var token: llama_token
        /// 根からの深さ(根は 0)
        var depth: Int
        /// 親の節の番号(根は nil)
        var parent: Int?
        /// この節を通る候補の番号(受け入れた順に 0 から)
        var candidates: [Int]
    }

    /// バッチに並べる順の節
    private(set) var nodes: [Node]
    /// 候補ごとの，根を除く節の番号の並び。木に入りきらなかった候補は nil
    private(set) var paths: [[Int]?]
    /// 木に入れた候補の数
    private(set) var acceptedCount: Int
    /// 候補の番号(元の並び)から，木の中の候補の番号(受け入れた順)への対応
    private(set) var acceptedIndices: [Int?]

    /// - Parameters:
    ///   - rootToken: プロンプトの最後のトークン
    ///   - candidates: 候補のトークン列(前ほど優先して入れる)
    ///   - maximumNodes: 根を含む節の数の上限(1 回の推論で評価するトークンの数)
    ///   - maximumCandidates: 入れる候補の数の上限(系列 ID の数)
    init(rootToken: llama_token, candidates: [[llama_token]], maximumNodes: Int, maximumCandidates: Int) {
        var nodes = [Node(token: rootToken, depth: 0, parent: nil, candidates: [])]
        var children: [Int: [llama_token: Int]] = [:]
        var paths: [[Int]?] = []
        var acceptedIndices: [Int?] = []
        var accepted = 0
        for tokens in candidates {
            guard accepted < maximumCandidates else {
                paths.append(nil)
                acceptedIndices.append(nil)
                continue
            }
            // 既にある節をたどり，足りない節の数を数えてから入れる(入りきらない候補は途中まで入れない)
            var current = 0
            var matched = 0
            for token in tokens {
                guard let next = children[current]?[token] else { break }
                current = next
                matched += 1
            }
            guard nodes.count + tokens.count - matched <= maximumNodes else {
                paths.append(nil)
                acceptedIndices.append(nil)
                continue
            }
            let index = accepted
            accepted += 1
            nodes[0].candidates.append(index)
            var path: [Int] = []
            current = 0
            for (depth, token) in tokens.enumerated() {
                if let next = children[current]?[token] {
                    current = next
                } else {
                    nodes.append(Node(token: token, depth: depth + 1, parent: current, candidates: []))
                    children[current, default: [:]][token] = nodes.count - 1
                    current = nodes.count - 1
                }
                nodes[current].candidates.append(index)
                path.append(current)
            }
            paths.append(path)
            acceptedIndices.append(index)
        }
        self.nodes = nodes
        self.paths = paths
        self.acceptedCount = accepted
        self.acceptedIndices = acceptedIndices
    }

    /// 節ごとの「次のトークンの対数確率」を引く関数から，候補ごとの log P(候補のトークン列, 終わり | プロンプト) を求める。
    /// - Parameter logProbability: (節の番号, トークン) → その節の出力での対数確率
    func scores(endToken: llama_token, logProbability: (Int, llama_token) -> Float) -> [Float?] {
        self.paths.map { path in
            guard let path else {
                return nil
            }
            var score: Float = 0
            var parent = 0
            for node in path {
                score += logProbability(parent, self.nodes[node].token)
                parent = node
            }
            return score + logProbability(parent, endToken)
        }
    }
}

/// ロジットの log Σ exp を求める。語彙の全体を節の数だけ回すので，SIMD で exp を近似して速くする(相対誤差は 1e-6 程度)
enum ZenzLogSumExp {
    static func compute(_ values: UnsafePointer<Float>, count: Int) -> Float {
        guard count > 0 else {
            return -.infinity
        }
        let width = 8
        var index = 0
        var maximumVector = SIMD8<Float>(repeating: -.infinity)
        while index + width <= count {
            maximumVector = pointwiseMax(maximumVector, UnsafeRawPointer(values + index).loadUnaligned(as: SIMD8<Float>.self))
            index += width
        }
        var maximum = maximumVector.max()
        while index < count {
            maximum = max(maximum, values[index])
            index += 1
        }
        guard maximum.isFinite else {
            return maximum
        }
        let shift = SIMD8<Float>(repeating: maximum)
        var sumVector = SIMD8<Float>(repeating: 0)
        index = 0
        while index + width <= count {
            sumVector += self.exp(UnsafeRawPointer(values + index).loadUnaligned(as: SIMD8<Float>.self) - shift)
            index += width
        }
        var sum = sumVector.sum()
        while index < count {
            sum += self.exp(SIMD8<Float>(repeating: values[index] - maximum))[0]
            index += 1
        }
        return maximum + logf(sum)
    }

    /// x ≤ 0 の exp。exp(x) = 2^k · exp(r)(k = round(x / ln 2)，|r| ≤ ln 2 / 2)に分け，exp(r) を 7 次の多項式で求める
    @inline(__always)
    static func exp(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let clamped = x.clamped(lowerBound: SIMD8(repeating: -87), upperBound: SIMD8(repeating: 0))
        let k = (clamped * 1.442_695_04).rounded(.toNearestOrEven)
        // ln 2 を 2 つに分けて引き，丸めの誤差を抑える
        let r = clamped - k * 0.693_145_751_953_125 - k * 1.428_606_765_330_187e-6
        var p = SIMD8<Float>(repeating: 1.0 / 5040)
        p = p * r + 1.0 / 720
        p = p * r + 1.0 / 120
        p = p * r + 1.0 / 24
        p = p * r + 1.0 / 6
        p = p * r + 0.5
        p = p * r + 1
        p = p * r + 1
        // 2^k を浮動小数点数のビットで作る(k ≥ -126 なので正規化数に収まる)
        let exponent = (SIMD8<Int32>(k) &+ 127) &<< 23
        return p * unsafeBitCast(exponent, to: SIMD8<Float>.self)
    }
}
