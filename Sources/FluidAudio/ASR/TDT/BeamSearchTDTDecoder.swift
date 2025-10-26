import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
internal struct BeamSearchTDTDecoder {
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "TDT.Beam")
    private let config: ASRConfig
    private let beamWidth: Int

    init(config: ASRConfig, beamWidth: Int) {
        self.config = config
        self.beamWidth = beamWidth
    }

    /// Perform beam search decoding on a sequence of joint logits.
    /// - Parameters:
    ///   - jointLogits: Array where each element contains token and duration logits for a time step.
    ///   - initialState: Decoder state used to seed the hypotheses.
    /// - Returns: Best scoring hypothesis after processing all steps.
    func beamSearchDecoding(
        jointLogits: [MLMultiArray],
        initialState: TdtDecoderState
    ) throws -> TdtHypothesis {
        // Start with a single empty hypothesis
        var beam: [TdtHypothesis] = [TdtHypothesis(decState: initialState, lastToken: initialState.lastToken)]

        for (timeIdx, logits) in jointLogits.enumerated() {
            let (tokenLogits, durationLogits) = try splitLogits(
                logits, durationElements: config.tdtConfig.durationBins.count)
            let (_, duration) = try processDurationLogits(
                durationLogits, durationBins: config.tdtConfig.durationBins)
            let topTokens = topK(tokenLogits, k: beamWidth)

            var candidates: [TdtHypothesis] = []
            candidates.reserveCapacity(beamWidth * topTokens.count)

            for hypothesis in beam {
                for (token, score) in topTokens {
                    var newHyp = hypothesis
                    let stateCopy = try TdtDecoderState(from: hypothesis.decState ?? initialState)
                    updateHypothesis(
                        &newHyp,
                        token: token,
                        score: score,
                        duration: duration,
                        timeIdx: timeIdx,
                        decoderState: stateCopy
                    )
                    candidates.append(newHyp)
                }
            }

            candidates.sort { $0.score > $1.score }
            beam = Array(candidates.prefix(beamWidth))
        }

        return beam.max(by: { $0.score < $1.score }) ?? TdtHypothesis()
    }

    // MARK: - Helper Methods

    /// Split joint logits into token and duration components
    private func splitLogits(
        _ logits: MLMultiArray,
        durationElements: Int
    ) throws -> (
        tokenLogits: [Float], durationLogits: [Float]
    ) {
        let totalElements = logits.count
        let vocabSize = totalElements - durationElements
        guard totalElements >= durationElements, vocabSize > 0 else {
            throw ASRError.processingFailed("Logits dimension mismatch")
        }

        var tokenLogits = [Float](repeating: 0, count: vocabSize)
        var durationLogits = [Float](repeating: 0, count: durationElements)
        for i in 0..<vocabSize {
            tokenLogits[i] = logits[i].floatValue
        }
        for i in 0..<durationElements {
            durationLogits[i] = logits[vocabSize + i].floatValue
        }
        return (tokenLogits, durationLogits)
    }

    /// Process duration logits to get a duration value
    private func processDurationLogits(
        _ durationLogits: [Float],
        durationBins: [Int]
    ) throws -> (
        bestDuration: Int, duration: Int
    ) {
        guard let maxValue = durationLogits.max(),
            let maxIndex = durationLogits.firstIndex(of: maxValue)
        else {
            throw ASRError.processingFailed("Invalid duration logits")
        }
        let duration = durationBins[maxIndex]
        return (maxIndex, duration)
    }

    /// Update hypothesis with new token information
    private func updateHypothesis(
        _ hypothesis: inout TdtHypothesis,
        token: Int,
        score: Float,
        duration: Int,
        timeIdx: Int,
        decoderState: TdtDecoderState
    ) {
        hypothesis.ySequence.append(token)
        hypothesis.score += score
        hypothesis.timestamps.append(timeIdx)
        hypothesis.decState = decoderState
        hypothesis.lastToken = token

        if config.tdtConfig.includeTokenDuration {
            hypothesis.tokenDurations.append(duration)
        }
    }

    /// Return top-k values with their indices
    private func topK(_ values: [Float], k: Int) -> [(Int, Float)] {
        let sorted = values.enumerated().sorted { $0.element > $1.element }
        return Array(sorted.prefix(k)).map { ($0.offset, $0.element) }
    }
}
