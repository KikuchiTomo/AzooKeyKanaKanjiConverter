#if Zenzai || ZenzaiCPU
// Zenzai/ZenzaiCPU が有効でない場合、llama-mock.swift の実装が利用される
import llama
#endif
#if canImport(Darwin)
import Darwin
#endif
#if os(Windows) && (Zenzai || ZenzaiCPU)
import WinSDK
#endif

import Algorithms
import Foundation
import SwiftUtils

enum ZenzError: LocalizedError {
    case couldNotLoadModel(path: String)
    case couldNotLoadContext
    case couldNotLoadVocab

    var errorDescription: String? {
        switch self {
        case .couldNotLoadContext: return "failed to load context"
        case .couldNotLoadModel(path: let path): return "could not load model weight at \(path)"
        case .couldNotLoadVocab: return "failed to load vocab"
        }
    }
}

/// llama.cpp のバックエンドはプロセス単位で一度だけ初期化する。
///
/// `llama_backend_free()` は現在 MPI 用の後始末にしか使われないため、モデルや
/// コンテキストより先に解放される可能性があるプロセス終了時の明示解放は行わない。
private enum ZenzBackend {
    private static let initialized: Void = {
        #if os(Windows) && (Zenzai || ZenzaiCPU)
        // Windows の llama.cpp は CPU の命令ごとの ggml-cpu-*.dll を実行時に選んで読む(GGML_BACKEND_DL)。
        // 実行ファイルの場所ではなく llama.dll と同じ場所から読む(テストでは llama.dll が PATH の先にある)
        if let directory = ZenzBackend.llamaLibraryDirectory() {
            ggml_backend_load_all_from_path(directory)
        } else {
            ggml_backend_load_all()
        }
        #endif
        #if canImport(Darwin) && (Zenzai || ZenzaiCPU)
        // b9637 の Metal は，プロセスの終わりに残った residency set があると GGML_ASSERT で止まる(ggml_metal_rsets_free)。
        // モデルはプロセスの終わりまで持つので，residency set を使わない。利用者が環境変数で決めていればそれに従う
        setenv("GGML_METAL_NO_RESIDENCY", "1", 0)
        #endif
        #if ZenzaiCPU && canImport(Darwin)
        // b9637 ではバックエンドを登録するときに Metal も初期化される。CPU で推論するときは Metal を読まない
        setenv("GGML_DISABLE_METAL", "1", 0)
        #endif
        llama_backend_init()
    }()

    #if os(Windows) && (Zenzai || ZenzaiCPU)
    /// 読み込んだ llama.dll のあるフォルダ(UTF-8)
    private static func llamaLibraryDirectory() -> String? {
        guard let module = "llama.dll".withCString(encodedAs: UTF16.self, { GetModuleHandleW($0) }) else {
            return nil
        }
        var buffer = [WCHAR](repeating: 0, count: 32_768)
        let length = Int(GetModuleFileNameW(module, &buffer, DWORD(buffer.count)))
        guard length > 0, length < buffer.count else {
            return nil
        }
        let path = String(decoding: buffer[..<length], as: UTF16.self)
        guard let separator = path.lastIndex(where: { $0 == "\\" || $0 == "/" }) else {
            return nil
        }
        return String(path[..<separator])
    }
    #endif

    static func initializeIfNeeded() {
        _ = self.initialized
    }
}

private final class ZenzTokenArrayBox {
    init(_ tokens: [llama_token]) {
        self.tokens = tokens
    }

    let tokens: [llama_token]
}

struct ZenzResolvedConversionCacheKey: Equatable, Hashable {
    var input: [ComposingText.InputElement]
    var convertTarget: String
    var convertTargetCursorPosition: Int?
    var keyboardLanguage: KeyboardLanguage
    var versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode
    var prefixConstraint: Kana2Kanji.PrefixConstraint
    var inferenceLimit: Int
}

struct ZenzDraftConversionCacheKey: Equatable, Hashable {
    var input: [ComposingText.InputElement]
    var convertTarget: String
    var convertTargetCursorPosition: Int?
    var keyboardLanguage: KeyboardLanguage
    var versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode
    var prefixConstraint: Kana2Kanji.PrefixConstraint
}

struct ZenzDraftConversion {
    var resultPrevs: [RegisteredNode]
    var resultLatticeHead: ZenzResolvedLatticeHead
}

struct ZenzResolvedConversion {
    var resultPrevs: [RegisteredNode]
    var resultLatticeHead: ZenzResolvedLatticeHead
    var prefixConstraint: Kana2Kanji.PrefixConstraint
    var satisfyingCandidate: Candidate
}

struct ZenzResolvedLatticeNode: Hashable {
    var data: DicdataElement
    var range: Lattice.LatticeRange
}

final class ZenzResolvedLatticeHead: @unchecked Sendable {
    init(nodes: [ZenzResolvedLatticeNode]) {
        self.nodes = nodes
        self.fingerprint = nodes.hashValue
    }

    let nodes: [ZenzResolvedLatticeNode]
    let fingerprint: Int
}

private final class ZenzResolvedConversionCache: @unchecked Sendable {
    private struct Entry {
        var value: ZenzResolvedConversion
        var accessIndex: UInt64
    }

    init(capacity: Int) {
        self.capacity = capacity
    }

    func value(for key: ZenzResolvedConversionCacheKey) -> ZenzResolvedConversion? {
        self.lock.withLock {
            guard var entry = self.entries[key] else {
                return nil
            }
            self.accessIndex &+= 1
            entry.accessIndex = self.accessIndex
            self.entries[key] = entry
            return entry.value
        }
    }

    func insert(_ value: ZenzResolvedConversion, for key: ZenzResolvedConversionCacheKey) {
        var value = value
        self.lock.withLock {
            // 逐次入力では最大辞書長を超えた後の先頭候補が同一になる。
            // 不変配列を既存entryと共有し、prefixごとの重複保持を避ける。
            if let sharedHead = self.entries.values.lazy.map({
                $0.value.resultLatticeHead
            }).first(where: {
                $0.fingerprint == value.resultLatticeHead.fingerprint
                    && $0.nodes == value.resultLatticeHead.nodes
            }) {
                value.resultLatticeHead = sharedHead
            }
            self.accessIndex &+= 1
            self.entries[key] = Entry(value: value, accessIndex: self.accessIndex)
            if self.entries.count > self.capacity,
               let oldestKey = self.entries.min(by: {
                   $0.value.accessIndex < $1.value.accessIndex
               })?.key {
                self.entries.removeValue(forKey: oldestKey)
            }
        }
    }

    private let capacity: Int
    private var entries: [ZenzResolvedConversionCacheKey: Entry] = [:]
    private var accessIndex: UInt64 = 0
    private let lock = NSLock()
}

/// 同じ辞書状態・入力・制約から得た不変のdraft経路を共有するLRU cache。
/// 可変な探索状態を含むlattice本体は保持しない。
final class ZenzDraftConversionCache: @unchecked Sendable {
    private struct Entry {
        var value: ZenzDraftConversion
        var accessIndex: UInt64
    }

    init(capacity: Int) {
        self.capacity = capacity
    }

    func value(for key: ZenzDraftConversionCacheKey) -> ZenzDraftConversion? {
        self.lock.withLock {
            guard var entry = self.entries[key] else { return nil }
            self.accessIndex &+= 1
            entry.accessIndex = self.accessIndex
            self.entries[key] = entry
            return entry.value
        }
    }

    func insert(_ value: ZenzDraftConversion, for key: ZenzDraftConversionCacheKey) {
        var value = value
        self.lock.withLock {
            if let sharedHead = self.entries.values.lazy.map({
                $0.value.resultLatticeHead
            }).first(where: {
                $0.fingerprint == value.resultLatticeHead.fingerprint
                    && $0.nodes == value.resultLatticeHead.nodes
            }) {
                value.resultLatticeHead = sharedHead
            }
            self.accessIndex &+= 1
            self.entries[key] = Entry(value: value, accessIndex: self.accessIndex)
            if self.entries.count > self.capacity,
               let oldestKey = self.entries.min(by: {
                   $0.value.accessIndex < $1.value.accessIndex
               })?.key {
                self.entries.removeValue(forKey: oldestKey)
            }
        }
    }

    private let capacity: Int
    private var entries: [ZenzDraftConversionCacheKey: Entry] = [:]
    private var accessIndex: UInt64 = 0
    private let lock = NSLock()
}

/// 1つの`KanaKanjiConverter`内で再利用するZenzaiのメモ化キャッシュ。
///
/// モデルやnative contextとは所有者を分け、別Converter間では共有しない。一方、
/// `stopComposition()`は入力中の状態だけを終了するAPIなので、このキャッシュは
/// compositionを跨いで保持する。
final class ZenzaiMemoizationCache: @unchecked Sendable {
    init(
        evaluationCapacity: Int = 256,
        resolvedConversionCapacity: Int = 64,
        draftConversionCapacity: Int = 128,
        promptTokenCapacity: Int = 128
    ) {
        self.evaluationCache = ZenzEvaluationCache(capacity: evaluationCapacity)
        self.resolvedConversionCache = ZenzResolvedConversionCache(
            capacity: resolvedConversionCapacity
        )
        self.draftConversionCache = ZenzDraftConversionCache(
            capacity: draftConversionCapacity
        )
        self.evaluationPromptTokenCache.countLimit = promptTokenCapacity
    }

    func cachedEvaluation(for key: ZenzEvaluationCacheKey) -> CandidateEvaluationResult? {
        self.evaluationCache.value(for: key)
    }

    func cacheEvaluation(_ result: CandidateEvaluationResult, for key: ZenzEvaluationCacheKey) {
        self.evaluationCache.insert(result, for: key)
    }

    func cachedResolvedConversion(
        for key: ZenzResolvedConversionCacheKey
    ) -> ZenzResolvedConversion? {
        self.resolvedConversionCache.value(for: key)
    }

    func cacheResolvedConversion(
        _ value: ZenzResolvedConversion,
        for key: ZenzResolvedConversionCacheKey
    ) {
        self.resolvedConversionCache.insert(value, for: key)
    }

    func cachedDraftConversion(for key: ZenzDraftConversionCacheKey) -> ZenzDraftConversion? {
        self.draftConversionCache.value(for: key)
    }

    func cacheDraftConversion(
        _ value: ZenzDraftConversion,
        for key: ZenzDraftConversionCacheKey
    ) {
        self.draftConversionCache.insert(value, for: key)
    }

    func cachedEvaluationPromptTokens(for prompt: String) -> [llama_token]? {
        self.evaluationPromptTokenCache.object(forKey: prompt as NSString)?.tokens
    }

    func cacheEvaluationPromptTokens(_ tokens: [llama_token], for prompt: String) {
        self.evaluationPromptTokenCache.setObject(
            ZenzTokenArrayBox(tokens),
            forKey: prompt as NSString
        )
    }

    private let evaluationCache: ZenzEvaluationCache
    private let resolvedConversionCache: ZenzResolvedConversionCache
    private let draftConversionCache: ZenzDraftConversionCache
    private let evaluationPromptTokenCache = NSCache<NSString, ZenzTokenArrayBox>()
}

/// 読み取り専用のGGUFモデルを複数の推論コンテキスト間で共有する。
///
/// KV cacheなどの可変状態は`ZenzContext`側に残し、モデルの重みとvocabularyだけを
/// 共有することで、同じモデルを利用するConverterごとの再ロードを避ける。
private final class SharedZenzModel {
    init(path: String) throws {
        ZenzBackend.initializeIfNeeded()
        var modelParams = llama_model_default_params()
        modelParams.use_mmap = true
        #if ZenzaiCPU
        modelParams.n_gpu_layers = 0
        modelParams.split_mode = LLAMA_SPLIT_MODE_NONE
        guard let cpuDevice = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) else {
            debug("Could not find CPU backend")
            throw ZenzError.couldNotLoadModel(path: path)
        }
        // NULL終端のCPU-onlyデバイスリストを渡す。`n_gpu_layers = 0`だけでは
        // llama.cppがMetalデバイスも列挙し、context生成時に初期化してしまう。
        var devices = [cpuDevice, nil]
        let loadedModel = devices.withUnsafeMutableBufferPointer { buffer in
            modelParams.devices = buffer.baseAddress
            return llama_model_load_from_file(path, modelParams)
        }
        #else
        let loadedModel = llama_model_load_from_file(path, modelParams)
        #endif
        guard let model = loadedModel else {
            debug("Could not load model at \(path)")
            throw ZenzError.couldNotLoadModel(path: path)
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_model_free(model)
            debug("Could not load vocab!")
            throw ZenzError.couldNotLoadVocab
        }
        self.model = model
        self.vocab = vocab
    }

    deinit {
        llama_model_free(self.model)
    }

    let model: OpaquePointer
    let vocab: OpaquePointer
}

/// 同時に強参照するモデルを1個に制限する、プロセス共通のモデルキャッシュ。
///
/// `NSCache`にすることでメモリプレッシャー時にはモデルを解放できる。呼び出し側の
/// `ZenzContext`もモデルを強参照するため、使用中のモデルが解放されることはない。
private final class SharedZenzModelCache: @unchecked Sendable {
    static let shared = SharedZenzModelCache()

    private init() {
        self.cache.countLimit = 1
    }

    func model(path: String) throws -> SharedZenzModel {
        try self.lock.withLock {
            let key = path as NSString
            if let cached = self.cache.object(forKey: key) {
                return cached
            }
            let model = try SharedZenzModel(path: path)
            self.cache.setObject(model, forKey: key)
            return model
        }
    }

    private let cache = NSCache<NSString, SharedZenzModel>()
    private let lock = NSLock()
}

final class ZenzContext {
    private let sharedModel: SharedZenzModel
    private var context: OpaquePointer
    private var batch: llama_batch
    private var prevInputBySeq: [llama_seq_id: [llama_token]] = [:]
    private var prevPromptBySeq: [llama_seq_id: String] = [:]

    private let n_len: Int32 = 512
    private let evalSeqId: llama_seq_id = 0
    private let inputPredictionSeqId: llama_seq_id = 1
    /// 検証木の候補に割り当てる系列 ID の始まり。評価と入力予測の系列(0・1)とは分ける
    private let candidateTreeFirstSeqId: llama_seq_id = 2
    /// 検証木を評価するバッチ。1 つのトークンに候補の数だけ系列 ID を付けるので，評価のバッチとは別に持つ(初めて使うときに作る)
    private var candidateTreeBatch: llama_batch?
    /// 検証木で 1 回に評価する候補の数の上限(系列 ID の数)。バッチの大きさを決めるので固定する
    static let candidateTreeMaximumCandidates = 32
    /// KV に置く系列の数(評価・入力予測・検証木の候補)
    private static let sequenceCount = 2 + candidateTreeMaximumCandidates

    private var memory: llama_memory_t {
        llama_get_memory(self.context)
    }

    private init(sharedModel: SharedZenzModel, context: OpaquePointer) {
        self.sharedModel = sharedModel
        self.context = context
        self.batch = llama_batch_init(512, 0, 1)
    }

    deinit {
        llama_batch_free(self.batch)
        if let candidateTreeBatch {
            llama_batch_free(candidateTreeBatch)
        }
        llama_free(context)
    }

    private static var ctx_params: llama_context_params {
        let n_threads = self.inferenceThreadCount
        debug("Using \(n_threads) threads")
        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = 512
        // 評価(0)・入力予測(1)・検証木の候補(2 から)の系列を 1 つの KV に置く。
        // 系列どうしはプロンプトの大部分を共有するので，KV を系列ごとに分けず，512 の容量を共有する
        ctx_params.n_seq_max = UInt32(Self.sequenceCount)
        ctx_params.kv_unified = true
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)
        ctx_params.n_batch = 512
        #if Zenzai || ZenzaiCPU
        ctx_params.n_ubatch = 64
        #endif
        // 推論時間は呼び出し側で計測する。llama.cpp内部の統計更新は不要。
        ctx_params.no_perf = true
        return ctx_params
    }

    /// 対話的な推論に使うCPUスレッド数を実行環境のトポロジーから決定する。
    ///
    /// Apple Siliconでは性能の異なるクラスタを跨ぐと短いdecodeのbarrier待ちが
    /// 増えるため、最高性能クラスタの物理コア数を利用する。それ以外の環境では
    /// OSへ応答性の余地を残しつつ、llama.cppの小規模モデルで過剰並列にならない
    /// よう8スレッドを上限とする。
    private static var inferenceThreadCount: Int {
        let activeProcessorCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        #if canImport(Darwin)
        let performanceCoreCount = self.sysctlInt32("hw.perflevel0.physicalcpu")
        #else
        let performanceCoreCount: Int? = nil
        #endif
        return self.selectInferenceThreadCount(
            activeProcessorCount: activeProcessorCount,
            performanceCoreCount: performanceCoreCount
        )
    }

    /// DarwinではmacOS/iOSともに最高性能クラスタを優先し、sysctl値を取得できない
    /// 端末やDarwin以外では論理CPU数から安全なフォールバックを選ぶ。
    static func selectInferenceThreadCount(
        activeProcessorCount: Int,
        performanceCoreCount: Int?
    ) -> Int {
        let activeProcessorCount = max(1, activeProcessorCount)
        if let performanceCoreCount,
           performanceCoreCount > 0,
           performanceCoreCount <= activeProcessorCount {
            return performanceCoreCount
        }
        let reservedProcessorCount = if activeProcessorCount >= 8 {
            2
        } else if activeProcessorCount >= 4 {
            1
        } else {
            0
        }
        return max(1, min(8, activeProcessorCount - reservedProcessorCount))
    }

    #if canImport(Darwin)
    private static func sysctlInt32(_ name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0, size == MemoryLayout<Int32>.size else {
            return nil
        }
        return Int(value)
    }
    #endif

    static func createContext(path: String) throws -> ZenzContext {
        let sharedModel = try SharedZenzModelCache.shared.model(path: path)
        var params = ctx_params
        #if ZenzaiCPU
        // CPU 専用: KV / KQV 等の GPU オフロードを完全に無効化
        params.offload_kqv = false
        #endif
        let context = llama_init_from_model(sharedModel.model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }

        return ZenzContext(sharedModel: sharedModel, context: context)
    }

    func resetContext() throws {
        llama_free(self.context)
        var params = Self.ctx_params
        #if ZenzaiCPU
        params.offload_kqv = false
        #endif
        let context = llama_init_from_model(self.sharedModel.model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }
        self.context = context
        self.prevInputBySeq = [:]
        self.prevPromptBySeq = [:]
    }

    private func getLogits(tokens: [llama_token], logits_start_index: Int = 0, seqId: llama_seq_id = 0) -> UnsafeMutablePointer<Float>? {
        let currentPrevInput = self.prevInputBySeq[seqId] ?? []
        var effectivePrevInput = currentPrevInput

        // Try to copy KV cache from the other sequence if it gives a longer prefix match.
        let otherSeqId: llama_seq_id? = if seqId == evalSeqId {
            inputPredictionSeqId
        } else if seqId == inputPredictionSeqId {
            evalSeqId
        } else {
            nil
        }
        if let otherSeqId, let otherPrevInput = self.prevInputBySeq[otherSeqId] {
            let currentPrefix = currentPrevInput.commonPrefix(with: tokens).count
            let otherPrefix = otherPrevInput.commonPrefix(with: tokens).count
            if otherPrefix > currentPrefix {
                let copiedPrefixCount = min(otherPrefix, logits_start_index)
                if copiedPrefixCount > 0 {
                    llama_memory_seq_rm(self.memory, seqId, 0, -1)
                    llama_memory_seq_cp(self.memory, otherSeqId, seqId, 0, llama_pos(copiedPrefixCount))
                    effectivePrevInput = otherPrevInput
                }
            }
        }

        // Manage KV cache: remove entries that differ from previous input
        let prefixCacheCount: Int
        do {
            let pos_max = llama_memory_seq_pos_max(self.memory, seqId)
            debug("pos max:", pos_max, "prevInput count:", effectivePrevInput.count, "tokens count:", tokens.count)
            let commonTokens = effectivePrevInput.commonPrefix(with: tokens)
            // Remove KV cache from position commonTokens.count onwards to recompute divergent part
            // removed range: [llama_pos(commonTokens.count), inf)
            prefixCacheCount = min(commonTokens.count, logits_start_index)
            llama_memory_seq_rm(self.memory, seqId, llama_pos(prefixCacheCount), -1)
            debug("new pos max:", llama_memory_seq_pos_max(self.memory, seqId), "commonTokens:", commonTokens.count)
        }
        self.batch.n_tokens = 0
        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens.count + (Int(n_len) - tokens.count)
        if n_kv_req > n_ctx {
            debug("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }
        for i in tokens.indices.dropFirst(prefixCacheCount) {
            llama_batch_add(&self.batch, tokens[i], Int32(i), [seqId], logits: logits_start_index <= i)
        }
        // 評価
        if llama_decode(context, self.batch) != 0 {
            debug("llama_decode() failed")
            return nil
        }
        // update cached input for next call (for KV cache management)
        self.prevInputBySeq[seqId] = tokens
        return llama_get_logits(context)
    }

    /// 候補のトークン列を検証木にまとめ，1 回の `llama_decode` で全候補の log P(候補, 出力の終わり | プロンプト) を求める。
    ///
    /// プロンプトの最後のトークンを除いた部分は評価の系列(0)の KV を使い回し，候補ごとの系列(2 から)に写す。
    /// 木の節はすべてロジットを出し，候補のトークンごとの対数確率と，最後の節での「出力の終わり」の対数確率を足す。
    /// 評価のあとで候補の系列の KV は消す。評価の系列の KV は，前の評価と食い違う部分だけをプロンプトに置き換える(食い違った続きは次の評価で求め直される)。
    /// KV に空きが無いときは推論せず，すべて nil を返す。
    /// - Returns: 候補ごとの点数(木に入りきらなかった候補と，推論に失敗したときは nil)，評価した木のトークンの数(根を含む節の数)，
    ///   KV に無かったので評価し直したプロンプトのトークンの数
    func scoreCandidateTree(
        promptTokens: [llama_token],
        candidates: [[llama_token]],
        maximumNodes: Int,
        maximumCandidates: Int
    ) -> (scores: [Float?], evaluatedTokens: Int, decodedPromptTokens: Int) {
        var failure = (scores: candidates.map { _ -> Float? in nil }, evaluatedTokens: 0, decodedPromptTokens: 0)
        guard let rootToken = promptTokens.last else {
            return failure
        }
        let head = Array(promptTokens.dropLast())
        let nodeLimit = min(maximumNodes, Int(self.n_len) - head.count)
        let tree = ZenzCandidateTree(
            rootToken: rootToken,
            candidates: candidates,
            maximumNodes: nodeLimit,
            maximumCandidates: min(maximumCandidates, Self.candidateTreeMaximumCandidates)
        )
        guard tree.acceptedCount > 0 else {
            return failure
        }

        // 1. プロンプトの最後を除いた部分を評価の系列に置く(前の評価と共通の先頭は使い回す)
        let previous = self.prevInputBySeq[evalSeqId] ?? []
        let common = previous.commonPrefix(with: head).count
        let decodedPromptTokens = head.count - common
        failure.decodedPromptTokens = decodedPromptTokens
        if common < head.count {
            llama_memory_seq_rm(self.memory, evalSeqId, llama_pos(common), -1)
            self.prevInputBySeq[evalSeqId] = Array(head.prefix(common))
            self.batch.n_tokens = 0
            for i in common ..< head.count {
                llama_batch_add(&self.batch, head[i], Int32(i), [evalSeqId], logits: false)
            }
            if llama_decode(context, self.batch) != 0 {
                debug("llama_decode() failed while placing the prompt of the candidate tree")
                return failure
            }
            self.prevInputBySeq[evalSeqId] = head
        }

        // 2. 候補ごとの系列にプロンプトを写す(KV を足さずにセルに系列 ID を足すだけ)
        let seqIds = (0 ..< tree.acceptedCount).map { self.candidateTreeFirstSeqId + llama_seq_id($0) }
        for seqId in seqIds {
            llama_memory_seq_rm(self.memory, seqId, -1, -1)
            if !head.isEmpty {
                llama_memory_seq_cp(self.memory, evalSeqId, seqId, 0, llama_pos(head.count))
            }
        }
        defer {
            for seqId in seqIds {
                llama_memory_seq_rm(self.memory, seqId, -1, -1)
            }
        }

        // 3. 木の節をまとめて 1 回で評価する
        // 大きさは n_len(512)と candidateTreeMaximumCandidates(32)に合わせる
        var treeBatch = self.candidateTreeBatch ?? llama_batch_init(512, 0, 32)
        self.candidateTreeBatch = treeBatch
        treeBatch.n_tokens = 0
        for node in tree.nodes {
            llama_batch_add(
                &treeBatch,
                node.token,
                Int32(head.count + node.depth),
                node.candidates.map { self.candidateTreeFirstSeqId + llama_seq_id($0) },
                logits: true
            )
        }
        if llama_decode(context, treeBatch) != 0 {
            debug("llama_decode() failed for the candidate tree")
            return failure
        }

        // 4. 節ごとに log softmax の正規化項を求め，候補ごとに対数確率を足す
        let vocabSize = self.vocabSize
        var normalizers: [Float] = []
        normalizers.reserveCapacity(tree.nodes.count)
        var rows: [UnsafeMutablePointer<Float>] = []
        rows.reserveCapacity(tree.nodes.count)
        for index in tree.nodes.indices {
            guard let row = llama_get_logits_ith(context, Int32(index)) else {
                return failure
            }
            rows.append(row)
            normalizers.append(ZenzLogSumExp.compute(row, count: vocabSize))
        }
        let scores = tree.scores(endToken: self.endOfOutputToken) { node, token in
            rows[node][Int(token)] - normalizers[node]
        }
        return (scores, tree.nodes.count, decodedPromptTokens)
    }

    /// 出力の終わりを表すトークン。zenz は語彙の「</s>」で出力を終える(GGUF の eos は別の番号を指していることがある)。
    /// 「</s>」が無ければ語彙の eos を使う
    private lazy var endOfOutputToken: llama_token = {
        let count = min(self.vocabSize, 16)
        for token in 0 ..< count {
            if let text = llama_vocab_get_text(self.sharedModel.vocab, llama_token(token)), String(cString: text) == "</s>" {
                return llama_token(token)
            }
        }
        return self.eosToken
    }()

    func previousEvaluationPrompt() -> String {
        self.prevPromptBySeq[evalSeqId] ?? ""
    }

    func setPreviousEvaluationPrompt(_ prompt: String) {
        self.prevPromptBySeq[evalSeqId] = prompt
    }

    func normalizeForModel(_ text: String) -> String {
        self.preprocessText(text: text)
    }

    func encodeEvaluationPrompt(
        _ prompt: String,
        memoizationCache: ZenzaiMemoizationCache
    ) -> [llama_token] {
        if let cached = memoizationCache.cachedEvaluationPromptTokens(for: prompt) {
            return cached
        }
        let tokens = self.encode(prompt, addBOS: true, addEOS: false)
        memoizationCache.cacheEvaluationPromptTokens(tokens, for: prompt)
        return tokens
    }

    func encode(_ text: String, addBOS: Bool, addEOS: Bool = false) -> [llama_token] {
        self.tokenize(text: self.preprocessText(text: text), add_bos: addBOS, add_eos: addEOS)
    }

    func encodeRaw(_ text: String, addBOS: Bool, addEOS: Bool = false) -> [llama_token] {
        self.tokenize(text: text, add_bos: addBOS, add_eos: addEOS)
    }

    func evaluationLogits(tokens: [llama_token], startOffset: Int) -> UnsafeMutablePointer<Float>? {
        self.getLogits(tokens: tokens, logits_start_index: startOffset, seqId: evalSeqId)
    }

    func inputPredictionLogits(tokens: [llama_token], startOffset: Int) -> UnsafeMutablePointer<Float>? {
        self.getLogits(tokens: tokens, logits_start_index: startOffset, seqId: inputPredictionSeqId)
    }

    var vocabSize: Int {
        Int(llama_vocab_n_tokens(self.sharedModel.vocab))
    }

    var eosToken: llama_token {
        llama_vocab_eos(self.sharedModel.vocab)
    }

    func decodeTokens(_ tokens: [llama_token]) -> String {
        let cchars: [CChar] = tokens.flatMap(self.tokenToPiece)
        let data = Data(cchars.map { UInt8(bitPattern: $0) })
        return String(data: data, encoding: .utf8) ?? ""
    }

    private func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], logits: Bool) {
        batch.token   [Int(batch.n_tokens)] = id
        batch.pos     [Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
        batch.n_tokens += 1
    }

    private func preprocessText(text: String) -> String {
        // replace space into ideographic space (\u3000) for zenz tokenizer
        // replace newline into null for zenz tokenizer
        text.replacingOccurrences(of: " ", with: "\u{3000}").replacingOccurrences(of: "\n", with: "")
    }
    private func tokenize(text: String, add_bos: Bool, add_eos: Bool = false) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(self.sharedModel.vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)
        var swiftTokens: [llama_token] = if tokenCount < 0 {
            [llama_vocab_bos(self.sharedModel.vocab)]
        } else {
            (0..<tokenCount).map {tokens[Int($0)]}
        }
        tokens.deallocate()
        if add_eos {
            swiftTokens.append(llama_vocab_eos(self.sharedModel.vocab))
        }
        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    func tokenToPiece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(self.sharedModel.vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(self.sharedModel.vocab, token, newResult, Int32(-nTokens), 0, false)
            let bufferPointer: UnsafeBufferPointer<Int8> = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer: UnsafeBufferPointer<Int8> = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}
