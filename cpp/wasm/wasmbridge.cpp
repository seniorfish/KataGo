#include "../wasm/wasmbridge.h"

#ifdef __EMSCRIPTEN__

#include <emscripten/threading_primitives.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <streambuf>

using namespace std;

namespace WasmBridge {

namespace {

// Ring capacity: GTP lines (including kata-analyze's position JSON) are a few KB; 512KB is plenty.
constexpr int32_t kRingCapacity = 512 * 1024;

BridgeCtrl* s_ctrl = nullptr;
std::once_flag s_initFlag;
// The output ring's producer is C++ (multiple pthreads may write concurrently); a mutex
// protects whole-line writes.
std::mutex s_outMutex;

BridgeCtrl* ensureCtrl() {
  std::call_once(s_initFlag, []() {
    BridgeCtrl* ctrl = static_cast<BridgeCtrl*>(malloc(sizeof(BridgeCtrl)));
    if(ctrl == nullptr)
      abort();
    memset(ctrl, 0, sizeof(BridgeCtrl));

    ctrl->inCap = kRingCapacity;
    ctrl->inBase = static_cast<int32_t>(reinterpret_cast<intptr_t>(malloc(kRingCapacity)));
    ctrl->outCap = kRingCapacity;
    ctrl->outBase = static_cast<int32_t>(reinterpret_cast<intptr_t>(malloc(kRingCapacity)));
    if(ctrl->inBase == 0 || ctrl->outBase == 0)
      abort();

    s_ctrl = ctrl;
  });
  return s_ctrl;
}

// Output ring: writes a whole line at the tail and advances outTail once, so the JS side
// only ever sees complete lines.
void writeOutputBytes(const char* data, size_t len, BridgeCtrl* ctrl) {
  const size_t need = len + 1;  // +1 for the trailing '\n'
  // Wait for space (SPSC: space = cap - (tail - head) - 1)
  for(;;) {
    const int32_t head = emscripten_atomic_load_u32(&ctrl->outHead);
    const int32_t tail = emscripten_atomic_load_u32(&ctrl->outTail);
    const int32_t space = ctrl->outCap - (tail - head) - 1;
    if(space >= static_cast<int32_t>(need))
      break;
    if(emscripten_atomic_load_u32(&ctrl->shutdownFlag) != 0)
      return;  // shutting down, drop
    emscripten_futex_wait(&ctrl->outHead, head, INFINITY);
  }

  const int32_t base = emscripten_atomic_load_u32(&ctrl->outTail);
  char* ring = reinterpret_cast<char*>(static_cast<intptr_t>(ctrl->outBase));
  for(size_t i = 0; i < len; i++)
    ring[(base + static_cast<int32_t>(i)) % ctrl->outCap] = data[i];
  ring[(base + static_cast<int32_t>(len)) % ctrl->outCap] = '\n';

  emscripten_atomic_store_u32(&ctrl->outTail, base + static_cast<int32_t>(need));
  emscripten_futex_wake(&ctrl->outTail, 1);
}

}  // namespace

void ensureInit() {
  ensureCtrl();
}

intptr_t getBridgeOffsets() {
  return reinterpret_cast<intptr_t>(ensureCtrl());
}

bool readGtpLine(std::string& line) {
  BridgeCtrl* ctrl = ensureCtrl();
  // Wait for input (SPSC: JS advancing tail means a complete line is available)
  for(;;) {
    const int32_t head = emscripten_atomic_load_u32(&ctrl->inHead);
    const int32_t tail = emscripten_atomic_load_u32(&ctrl->inTail);
    if(head != tail)
      break;
    if(emscripten_atomic_load_u32(&ctrl->shutdownFlag) != 0)
      return false;
    emscripten_futex_wait(&ctrl->inTail, tail, INFINITY);
  }

  // Read one line byte by byte (up to '\n' or until data runs out), advancing inHead
  line.clear();
  char* ring = reinterpret_cast<char*>(static_cast<intptr_t>(ctrl->inBase));
  for(;;) {
    const int32_t head = emscripten_atomic_load_u32(&ctrl->inHead);
    const int32_t tail = emscripten_atomic_load_u32(&ctrl->inTail);
    if(head >= tail)
      break;
    const char c = ring[head % ctrl->inCap];
    emscripten_atomic_store_u32(&ctrl->inHead, head + 1);
    if(c == '\n')
      break;
    line.push_back(c);
  }
  return true;
}

void writeOutputLine(const std::string& line) {
  BridgeCtrl* ctrl = ensureCtrl();
  std::lock_guard<std::mutex> lock(s_outMutex);
  writeOutputBytes(line.data(), line.size(), ctrl);
}

void shutdown() {
  BridgeCtrl* ctrl = ensureCtrl();
  emscripten_atomic_store_u32(&ctrl->shutdownFlag, 1);
  emscripten_futex_wake(&ctrl->inTail, 0x7fffffff);
  emscripten_futex_wake(&ctrl->outHead, 0x7fffffff);
  emscripten_futex_wake(&ctrl->inHead, 0x7fffffff);
}

// ---- NN inference bridge ------------------------------------------------------------------

void setupInference(
  int numInputChannels, int numInputGlobalChannels, int numInputMetaChannels,
  int numPolicyChannels, int numValueChannels, int numScoreValueChannels, int numOwnershipChannels,
  int nnXLen, int nnYLen, int maxBatchSize) {
  BridgeCtrl* ctrl = ensureCtrl();
  if(ctrl->inSpatialOff != 0)
    return;  // already initialized (idempotent when ComputeHandle is created for multiple threads)

  ctrl->metaHasData = (numInputMetaChannels > 0) ? 1 : 0;
  ctrl->inSpatialElts = numInputChannels * nnXLen * nnYLen;
  ctrl->inGlobalElts = numInputGlobalChannels;
  ctrl->inMaskElts = nnXLen * nnYLen;
  ctrl->inMetaElts = numInputMetaChannels;
  ctrl->outPolicyPassElts = numPolicyChannels;
  ctrl->outPolicyElts = numPolicyChannels * nnXLen * nnYLen;
  ctrl->outValueElts = numValueChannels;
  ctrl->outScoreElts = numScoreValueChannels;
  ctrl->outOwnElts = numOwnershipChannels * nnXLen * nnYLen;

  const int inTotal = (ctrl->inSpatialElts + ctrl->inGlobalElts + ctrl->inMaskElts + ctrl->inMetaElts) * maxBatchSize;
  const int outTotal =
    (ctrl->outPolicyPassElts + ctrl->outPolicyElts + ctrl->outValueElts + ctrl->outScoreElts + ctrl->outOwnElts) *
    maxBatchSize;
  float* inBuf = static_cast<float*>(malloc(inTotal * sizeof(float)));
  float* outBuf = static_cast<float*>(malloc(outTotal * sizeof(float)));
  if(inBuf == nullptr || outBuf == nullptr)
    abort();
  memset(inBuf, 0, inTotal * sizeof(float));
  memset(outBuf, 0, outTotal * sizeof(float));

  int base = 0;
  ctrl->inSpatialOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(inBuf + base));
  base += ctrl->inSpatialElts * maxBatchSize;
  ctrl->inGlobalOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(inBuf + base));
  base += ctrl->inGlobalElts * maxBatchSize;
  ctrl->inMaskOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(inBuf + base));
  base += ctrl->inMaskElts * maxBatchSize;
  ctrl->inMetaOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(inBuf + base));

  base = 0;
  ctrl->outPolicyPassOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(outBuf + base));
  base += ctrl->outPolicyPassElts * maxBatchSize;
  ctrl->outPolicyOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(outBuf + base));
  base += ctrl->outPolicyElts * maxBatchSize;
  ctrl->outValueOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(outBuf + base));
  base += ctrl->outValueElts * maxBatchSize;
  ctrl->outScoreOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(outBuf + base));
  base += ctrl->outScoreElts * maxBatchSize;
  ctrl->outOwnOff = static_cast<int32_t>(reinterpret_cast<intptr_t>(outBuf + base));
}

float* inputRegionPtr(int region) {
  BridgeCtrl* ctrl = ensureCtrl();
  switch(region) {
    case INPUT_SPATIAL: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->inSpatialOff));
    case INPUT_GLOBAL: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->inGlobalOff));
    case INPUT_MASK: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->inMaskOff));
    case INPUT_META: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->inMetaOff));
  }
  return nullptr;
}

float* outputRegionPtr(int region) {
  BridgeCtrl* ctrl = ensureCtrl();
  switch(region) {
    case OUT_POLICY_PASS: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->outPolicyPassOff));
    case OUT_POLICY: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->outPolicyOff));
    case OUT_VALUE: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->outValueOff));
    case OUT_SCORE: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->outScoreOff));
    case OUT_OWNERSHIP: return reinterpret_cast<float*>(static_cast<intptr_t>(ctrl->outOwnOff));
  }
  return nullptr;
}

int inputRegionElts(int region) {
  BridgeCtrl* ctrl = ensureCtrl();
  switch(region) {
    case INPUT_SPATIAL: return ctrl->inSpatialElts;
    case INPUT_GLOBAL: return ctrl->inGlobalElts;
    case INPUT_MASK: return ctrl->inMaskElts;
    case INPUT_META: return ctrl->inMetaElts;
  }
  return 0;
}

int outputRegionElts(int region) {
  BridgeCtrl* ctrl = ensureCtrl();
  switch(region) {
    case OUT_POLICY_PASS: return ctrl->outPolicyPassElts;
    case OUT_POLICY: return ctrl->outPolicyElts;
    case OUT_VALUE: return ctrl->outValueElts;
    case OUT_SCORE: return ctrl->outScoreElts;
    case OUT_OWNERSHIP: return ctrl->outOwnElts;
  }
  return 0;
}

void runInference(int batchSize) {
  BridgeCtrl* ctrl = ensureCtrl();
  // Request/response use monotonic counters (immune to missed/spurious wakeups). The NN
  // worker (JS side) waits for inferReq to change, runs the model, writes outputs, then
  // increments inferResp.
  emscripten_atomic_store_u32(&ctrl->inferBatch, batchSize);
  const uint32_t req = emscripten_atomic_load_u32(&ctrl->inferReq) + 1;
  emscripten_atomic_store_u32(&ctrl->inferReq, req);
  emscripten_futex_wake(&ctrl->inferReq, 1);

  // Block until the NN worker responds
  while(emscripten_atomic_load_u32(&ctrl->inferResp) < req) {
    if(emscripten_atomic_load_u32(&ctrl->shutdownFlag) != 0)
      return;
    emscripten_futex_wait(&ctrl->inferResp, emscripten_atomic_load_u32(&ctrl->inferResp), INFINITY);
  }
}

namespace {

// Line-buffered cout/cerr streambuf: writes a whole line to the output ring on '\n' or flush.
class RingOutBuf : public std::streambuf {
 public:
  explicit RingOutBuf() = default;

 protected:
  int_type overflow(int_type ch) override {
    if(ch != traits_type::eof()) {
      line_.push_back(static_cast<char>(ch));
      if(ch == '\n')
        flushLine();
    }
    return ch;
  }

  int sync() override {
    if(!line_.empty())
      flushLine();
    return 0;
  }

 private:
  void flushLine() {
    writeOutputLine(line_);
    line_.clear();
  }

  std::string line_;
};

}  // namespace

void installIO() {
  ensureInit();
  static RingOutBuf s_outBuf;
  std::cout.rdbuf(&s_outBuf);
  std::cerr.rdbuf(&s_outBuf);
}

}  // namespace WasmBridge

// C-linkage export (for EXPORTED_FUNCTIONS)
extern "C" intptr_t getBridgeOffsets() {
  return WasmBridge::getBridgeOffsets();
}

#endif  // __EMSCRIPTEN__
