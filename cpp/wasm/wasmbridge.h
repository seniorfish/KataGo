#ifndef KATAGO_WASM_BRIDGE_H_
#define KATAGO_WASM_BRIDGE_H_

// WASM bridge layer: lets the KataGo engine run under Emscripten + pthreads and
// communicate with the JS side (engineWorker / nnWorker) over shared memory
// (the wasm heap IS a SharedArrayBuffer) + Atomics.
//
// Responsibilities:
//   - GTP input: JS writes into the input ring -> readGtpLine() blocks on it (replaces std::cin)
//   - GTP output: installIO() swaps cout/cerr for a streambuf that writes the output ring -> JS drains back to the UI
//   - NN inference bridge: the NN server thread copies its inputs into the pinned region,
//     bumps the inferReq counter, then waits for inferResp (served by onnxruntime-web in a JS worker)
//
// The memory layout must stay exactly in sync with the Int32Array/Float32Array views
// used by the browser-side frontend's engine bridge.

#ifdef __EMSCRIPTEN__

#include <cstdint>
#include <string>

namespace WasmBridge {

// Control structure with a fixed layout in the engine heap (allocated once with malloc).
// Field order IS the memory layout; any change here must be mirrored in the frontend.
struct BridgeCtrl {
  // Input SPSC ring (JS -> engine), byte buffer
  int32_t inHead;   // read position (consumer = C++)
  int32_t inTail;   // write position (producer = JS)
  int32_t inCap;    // ring capacity (bytes)
  int32_t inBase;   // byte offset of the ring data area in the wasm heap
  // Output SPSC ring (engine -> JS)
  int32_t outHead;  // read position (consumer = JS)
  int32_t outTail;  // write position (producer = C++)
  int32_t outCap;
  int32_t outBase;
  // Inference request/response counters (monotonic counters, immune to missed/spurious wakeups)
  int32_t inferReq;
  int32_t inferResp;
  int32_t inferBatch;
  // Shutdown sentinel
  int32_t shutdownFlag;
  // Byte offsets and per-sample element counts of the pinned input float regions
  int32_t inSpatialOff, inGlobalOff, inMaskOff, inMetaOff;
  int32_t inSpatialElts, inGlobalElts, inMaskElts, inMetaElts;
  // Pinned output float regions
  int32_t outPolicyPassOff, outPolicyOff, outValueOff, outScoreOff, outOwnOff;
  int32_t outPolicyPassElts, outPolicyElts, outValueElts, outScoreElts, outOwnElts;
  int32_t metaHasData;
};

// Lazy initialization: allocates the ctrl struct + rings. Thread-safe (guarded by a once flag).
void ensureInit();

// Returns the byte offset of BridgeCtrl in the wasm heap (exported for JS-side views).
intptr_t getBridgeOffsets();

// Reads one line from the input ring (blocks until data is available or shutdown).
// Returns false when the engine should exit.
bool readGtpLine(std::string& line);

// Redirects cout/cerr to the output ring (line-buffered, internally mutex-protected).
void installIO();

// Writes one line to the output ring (used by installIO's streambuf and for debugging).
void writeOutputLine(const std::string& line);

// Sets the shutdown flag and wakes all waiters.
void shutdown();

// ---- NN inference bridge ----

// Pinned input/output float regions (in the wasm heap; offsets written into ctrl so the
// NN worker / JS side can read/write them with zero copies).
enum InputRegion : int { INPUT_SPATIAL = 0, INPUT_GLOBAL = 1, INPUT_MASK = 2, INPUT_META = 3 };
enum OutputRegion : int { OUT_POLICY_PASS = 0, OUT_POLICY = 1, OUT_VALUE = 2, OUT_SCORE = 3, OUT_OWNERSHIP = 4 };

// Allocates the pinned input/output buffers for the model's channel counts and writes the
// corresponding ctrl fields. Call once after the model is loaded.
void setupInference(
  int numInputChannels, int numInputGlobalChannels, int numInputMetaChannels,
  int numPolicyChannels, int numValueChannels, int numScoreValueChannels, int numOwnershipChannels,
  int nnXLen, int nnYLen, int maxBatchSize);

// Returns a float pointer to the pinned input/output region (sample n is at base + n*elts).
float* inputRegionPtr(int region);
float* outputRegionPtr(int region);
int inputRegionElts(int region);   // per-sample element count
int outputRegionElts(int region);

// Triggers one inference and blocks for the result (the NN worker runs the Atomics protocol).
void runInference(int batchSize);

}  // namespace WasmBridge

#endif  // __EMSCRIPTEN__
#endif  // KATAGO_WASM_BRIDGE_H_
