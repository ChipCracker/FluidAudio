import CoreML
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class BeamSearchTDTDecoderTests: XCTestCase {
    func testBeamSearchDecoding() throws {
        let config = ASRConfig.default
        let decoder = BeamSearchTDTDecoder(config: config, beamWidth: 2)

        // Each step has 4 token logits and 5 duration logits (total 9)
        let step1 = try MLMultiArray(shape: [9], dataType: .float32)
        // token logits
        step1[0] = 0.9
        step1[1] = 0.1
        step1[2] = 0.8
        step1[3] = 0.0
        // duration logits (choose index 1 -> duration 1)
        step1[4] = 0.1
        step1[5] = 0.9
        step1[6] = 0.1
        step1[7] = 0.1
        step1[8] = 0.1

        let step2 = try MLMultiArray(shape: [9], dataType: .float32)
        step2[0] = 0.7
        step2[1] = 0.4
        step2[2] = 0.6
        step2[3] = 0.0
        step2[4] = 0.1
        step2[5] = 0.9
        step2[6] = 0.1
        step2[7] = 0.1
        step2[8] = 0.1

        let initialState = try TdtDecoderState()
        let hypothesis = try decoder.beamSearchDecoding(
            jointLogits: [step1, step2],
            initialState: initialState
        )

        XCTAssertEqual(hypothesis.ySequence, [0, 0])
        XCTAssertEqual(hypothesis.timestamps, [0, 1])
        XCTAssertEqual(hypothesis.tokenDurations, [1, 1])
        XCTAssertEqual(hypothesis.lastToken, 0)
    }
}
