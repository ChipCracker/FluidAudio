public struct TdtConfig: Sendable {
    public let includeTokenDuration: Bool
    public let maxSymbolsPerStep: Int
    public let durationBins: [Int]
    public let blankId: Int
    /// Beam width for beam search decoding. Values > 1 enable beam search.
    public let beamWidth: Int

    public static let `default` = TdtConfig()

    public init(
        includeTokenDuration: Bool = true,
        maxSymbolsPerStep: Int = 10,
        durationBins: [Int] = [0, 1, 2, 3, 4],
        // Parakeet-TDT-0.6b-v3 uses 8192 regular tokens + blank token at index 8192
        blankId: Int = 8192,
        beamWidth: Int = 1
    ) {
        self.includeTokenDuration = includeTokenDuration
        self.maxSymbolsPerStep = maxSymbolsPerStep
        self.durationBins = durationBins
        self.blankId = blankId
        self.beamWidth = beamWidth
    }
}
