#if Zenzai || ZenzaiCPU
import llama
#endif
public import Foundation
import SwiftUtils

/// jinen(読みと左文脈から表記を出すかな漢字変換の言語モデル．Qwen3 系の GGUF)で変換し，与えた表記を採点する。
///
/// プロンプトは「[U+EE02 左文脈] U+EE00 読み(カタカナ) U+EE01」を NFKC にしたもの。出力は NFKC の表記。
/// 変換は 1 トークンずつ進められる(beginConversion と advance)。呼び出し側が打鍵の合間に少しずつ進めるため。
/// KV に置いたトークンを覚えておき，次の呼び出しでは共通の先頭を使い回す。ほかの呼び出しが挟まっても，足りない分を評価し直す。
/// 1 つのインスタンスは 1 つのスレッドから使う。
/// - Warning: このAPIは experimental であり、予告なく変更または削除される可能性があります。
public final class ExperimentalJinen {
    public struct Conversion: Sendable, Equatable {
        /// 出力の表記(NFKC)
        public var text: String
        /// log P(表記, 出力の終わり | 読み, 文脈)
        public var logProbability: Float
    }

    /// 途中まで進めた変換
    public struct ConversionProgress: Sendable {
        public enum State: Sendable, Equatable {
            case running
            /// 出し終えた。出力が空だったか，トークンの数の上限までに終わらなかったときは nil
            case finished(Conversion?)
            /// 推論に失敗した
            case failed
        }

        public fileprivate(set) var state: State = .running
        fileprivate var tokens: [Int32]
        fileprivate var output: [UInt8] = []
        fileprivate var logProbability: Float = 0
        fileprivate var remaining: Int
    }

    #if Zenzai || ZenzaiCPU
    private let model: OpaquePointer
    private let vocab: OpaquePointer
    private let context: OpaquePointer
    private var batch: llama_batch
    /// KV に置いたトークン(位置 0 から)
    private var placed: [llama_token] = []
    private let vocabularySize: Int
    private let endOfOutput: llama_token
    private static let contextLength: Int32 = 512

    /// GGUF を読む。読めなければ投げる
    public init(modelURL: URL) throws {
        ZenzBackend.initializeIfNeeded()
        var modelParams = llama_model_default_params()
        modelParams.use_mmap = true
        #if ZenzaiCPU
        modelParams.n_gpu_layers = 0
        #endif
        guard let model = llama_model_load_from_file(modelURL.path, modelParams) else {
            throw ZenzError.couldNotLoadModel(path: modelURL.path)
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_model_free(model)
            throw ZenzError.couldNotLoadVocab
        }
        var params = llama_context_default_params()
        params.n_ctx = UInt32(Self.contextLength)
        params.n_batch = UInt32(Self.contextLength)
        params.n_ubatch = 64
        params.n_seq_max = 1
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        params.n_threads = Int32(ZenzContext.inferenceThreadCount)
        params.n_threads_batch = Int32(ZenzContext.inferenceThreadCount)
        params.no_perf = true
        #if ZenzaiCPU
        params.offload_kqv = false
        #endif
        guard let context = llama_init_from_model(model, params) else {
            llama_model_free(model)
            throw ZenzError.couldNotLoadContext
        }
        self.model = model
        self.vocab = vocab
        self.context = context
        self.batch = llama_batch_init(Self.contextLength, 0, 1)
        self.vocabularySize = Int(llama_vocab_n_tokens(vocab))
        self.endOfOutput = llama_vocab_eos(vocab)
    }

    deinit {
        llama_batch_free(self.batch)
        llama_free(self.context)
        llama_model_free(self.model)
    }

    /// 読み(ひらがな)を貪欲法で変換し始める。プロンプトを作れないときは nil
    public func beginConversion(reading: String, leftContext: String, maximumLeftContextLength: Int = 40) -> ConversionProgress? {
        guard !reading.isEmpty, let (prompt, _) = self.prompt(reading: reading, leftContext: leftContext, maximumLeftContextLength: maximumLeftContextLength) else {
            return nil
        }
        return ConversionProgress(tokens: prompt, remaining: reading.count * 2 + 8)
    }

    /// 変換を 1 トークン進める(推論 1 回)。出し終えたら progress.state が finished になる
    public func advance(_ progress: inout ConversionProgress) {
        guard progress.state == .running else {
            return
        }
        guard progress.remaining > 0 else {
            progress.state = .finished(nil)
            return
        }
        progress.remaining -= 1
        guard let row = self.logits(after: progress.tokens) else {
            progress.state = .failed
            return
        }
        let normalizer = ZenzLogSumExp.compute(row, count: self.vocabularySize)
        var best = 0
        for index in 1 ..< self.vocabularySize where row[index] > row[best] {
            best = index
        }
        progress.logProbability += row[best] - normalizer
        if llama_vocab_is_eog(self.vocab, llama_token(best)) {
            let text = String(decoding: progress.output, as: UTF8.self)
            progress.state = .finished(text.isEmpty ? nil : Conversion(text: text, logProbability: progress.logProbability))
            return
        }
        progress.output += self.piece(llama_token(best))
        progress.tokens.append(llama_token(best))
    }

    /// 読み(ひらがな)を貪欲法で変換する。推論に失敗したとき，トークンの数の上限までに出力が終わらなかったときは nil
    public func convert(reading: String, leftContext: String, maximumLeftContextLength: Int = 40) -> Conversion? {
        guard var progress = self.beginConversion(reading: reading, leftContext: leftContext, maximumLeftContextLength: maximumLeftContextLength) else {
            return nil
        }
        while progress.state == .running {
            self.advance(&progress)
        }
        if case .finished(let conversion) = progress.state {
            return conversion
        }
        return nil
    }

    /// 表記ごとに log P(表記, 出力の終わり | 読み, 文脈) を求める。表記は NFKC にしてから測る。推論に失敗した表記は nil
    public func score(_ texts: [String], reading: String, leftContext: String, maximumLeftContextLength: Int = 40) -> [Float?] {
        guard !reading.isEmpty, let (prompt, promptText) = self.prompt(reading: reading, leftContext: leftContext, maximumLeftContextLength: maximumLeftContextLength) else {
            return texts.map { _ in nil }
        }
        return texts.map { text in
            let continuation = self.tokens(of: text.precomposedStringWithCompatibilityMapping, after: prompt, promptText: promptText)
            guard !continuation.isEmpty, prompt.count + continuation.count + 1 <= Int(Self.contextLength),
                  self.place(Array(prompt.dropLast())) else {
                return nil
            }
            let input = [prompt[prompt.count - 1]] + continuation
            guard self.decode(input, from: prompt.count - 1, allOutputs: true) else {
                return nil
            }
            var total: Float = 0
            for (index, target) in (continuation + [self.endOfOutput]).enumerated() {
                guard let row = llama_get_logits_ith(self.context, Int32(index)) else {
                    return nil
                }
                total += row[Int(target)] - ZenzLogSumExp.compute(row, count: self.vocabularySize)
            }
            return total
        }
    }

    /// プロンプトのトークンと文字列
    private func prompt(reading: String, leftContext: String, maximumLeftContextLength: Int) -> ([llama_token], String)? {
        let context = String(leftContext.suffix(maximumLeftContextLength))
        let text = ((context.isEmpty ? "" : "\u{EE02}" + context) + "\u{EE00}" + reading.toKatakana() + "\u{EE01}").precomposedStringWithCompatibilityMapping
        let prompt = self.tokenize(text, addSpecial: true)
        guard prompt.count >= 2, prompt.count + 1 < Int(Self.contextLength) else {
            return nil
        }
        return (prompt, text)
    }

    /// tokens の最後のトークンの次のロジット。最後を除いた部分は KV に置いたもの(共通の先頭)を使い回す
    private func logits(after tokens: [llama_token]) -> UnsafeMutablePointer<Float>? {
        guard let last = tokens.last, tokens.count < Int(Self.contextLength), self.place(Array(tokens.dropLast())),
              self.decode([last], from: tokens.count - 1, allOutputs: false) else {
            return nil
        }
        return llama_get_logits_ith(self.context, -1)
    }

    /// KV がちょうど tokens になるようにする(共通の先頭は残し，違う部分を消して評価し直す)
    private func place(_ tokens: [llama_token]) -> Bool {
        let common = self.placed.commonPrefix(with: tokens).count
        llama_memory_seq_rm(llama_get_memory(self.context), 0, llama_pos(common), -1)
        self.placed = Array(tokens.prefix(common))
        guard common < tokens.count else {
            return true
        }
        return self.decode(Array(tokens[common...]), from: common, allOutputs: false, logitsForLast: false)
    }

    /// トークンを position から並べて評価し，KV に置いたトークンを覚える。allOutputs ならすべてのトークンの，そうでなければ最後のトークンのロジットを出す
    private func decode(_ tokens: [llama_token], from position: Int, allOutputs: Bool, logitsForLast: Bool = true) -> Bool {
        guard !tokens.isEmpty, position == self.placed.count, position + tokens.count <= Int(Self.contextLength) else {
            return false
        }
        self.batch.n_tokens = 0
        for (index, token) in tokens.enumerated() {
            let slot = Int(self.batch.n_tokens)
            self.batch.token[slot] = token
            self.batch.pos[slot] = llama_pos(position + index)
            self.batch.n_seq_id[slot] = 1
            self.batch.seq_id[slot]![0] = 0
            self.batch.logits[slot] = (allOutputs || (logitsForLast && index == tokens.count - 1)) ? 1 : 0
            self.batch.n_tokens += 1
        }
        guard llama_decode(self.context, self.batch) == 0 else {
            // 途中まで置いたかわからないので，KV を空にする
            llama_memory_seq_rm(llama_get_memory(self.context), 0, 0, -1)
            self.placed = []
            return false
        }
        self.placed += tokens
        return true
    }

    /// prompt に続く text のトークン。続けて分けたものの先頭が prompt と一致しなければ，text だけを分ける
    private func tokens(of text: String, after prompt: [llama_token], promptText: String) -> [llama_token] {
        let joined = self.tokenize(promptText + text, addSpecial: true)
        if joined.count > prompt.count, Array(joined.prefix(prompt.count)) == prompt {
            return Array(joined.dropFirst(prompt.count))
        }
        return self.tokenize(text, addSpecial: false)
    }

    private func tokenize(_ text: String, addSpecial: Bool) -> [llama_token] {
        let capacity = text.utf8.count + 8
        var tokens = [llama_token](repeating: 0, count: capacity)
        let count = llama_tokenize(self.vocab, text, Int32(text.utf8.count), &tokens, Int32(capacity), addSpecial, true)
        return count > 0 ? Array(tokens.prefix(Int(count))) : []
    }

    private func piece(_ token: llama_token) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 64)
        let count = llama_token_to_piece(self.vocab, token, &buffer, Int32(buffer.count), 0, false)
        return count > 0 ? buffer.prefix(Int(count)).map { UInt8(bitPattern: $0) } : []
    }
    #else
    /// Zenzai の trait が無いビルドでは使えない
    public init(modelURL: URL) throws {
        throw ZenzError.couldNotLoadModel(path: modelURL.path)
    }

    public func beginConversion(reading: String, leftContext: String, maximumLeftContextLength: Int = 40) -> ConversionProgress? {
        nil
    }

    public func advance(_ progress: inout ConversionProgress) {
        progress.state = .failed
    }

    public func convert(reading: String, leftContext: String, maximumLeftContextLength: Int = 40) -> Conversion? {
        nil
    }

    public func score(_ texts: [String], reading: String, leftContext: String, maximumLeftContextLength: Int = 40) -> [Float?] {
        texts.map { _ in nil }
    }
    #endif
}
