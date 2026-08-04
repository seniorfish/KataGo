# ONNX input declaration order — verification

Investigation scripts for the `alignInputsToConsumptionOrder` workaround in
KataGo's ONNX backend (PR [#1222](https://github.com/lightvector/KataGo/pull/1222)).
The workaround reorders the internal ONNX graph's declared inputs from the master
default `(InputMask, InputSpatial, InputGlobal)` to `(InputSpatial, InputGlobal,
InputMask)` because the OpenVINO execution provider under ONNX Runtime misroutes
the mask tensor otherwise.

This repo branch is a fork-only add-on; it is intentionally *not* part of the PR.

## Background / what was questioned

The review asked three things about the workaround:

1. **API design** — why an opt-in boolean instead of always ordering unconditionally?
2. **"Consumption order" definition** — the comment claims consumption order is
   `InputSpatial (trunk conv) → InputMask (first applyMask) → InputGlobal (gpool)`.
   But in the `!requireExactNNLen` branch, `InputMask` is actually the *first*
   input referenced by the graph (the `ReduceSum`/`ReduceMean` mask features are
   emitted immediately after `addInput`). Does the fix still hold there?
3. **Fragility / vendor compliance** — is the EP violating the ONNX Runtime
   interface contract (bind inputs by name)? Is the workaround robust across
   `requireExactNNLen` and HumanSL variants?

## Key results

### 1. Static: real first-consumption order differs from the comment's claim

`python analyze_inputs.py` on the dumped models:

| branch | declared (workaround) | actual first-consumption order |
|---|---|---|
| `requireExactNNLen=true` | `Spatial, Global, Mask` | `Spatial (node0 Conv)`, `Global (node1 Conv)`, **`Mask` is a dead input (never consumed)** |
| `requireExactNNLen=false` | `Spatial, Global, Mask` | **`Mask (node0 ReduceSum)`**, `Spatial (node12 Conv)`, `Global (node13 Conv)` |

So in the `!requireExactNNLen` branch the mask is referenced first, exactly as the
review suspected. The workaround's "consumption order" is *not* the graph's
first-reference order.

### 2. Name-bound backends: declaration order is purely cosmetic

`python permute_and_run.py <model> CPUExecutionProvider`: all 6 permutations of
the 3 inputs run and produce **bit-identical outputs** on the CPU EP. The claim in
the code comment ("purely cosmetic for backends that bind inputs by name") holds
— for the CPU EP.

### 3. Official release line (ORT 1.24.1 + OpenVINO 2025.4) has no such bug

With `onnxruntime-openvino 1.24.1` + OpenVINO `2025.4.1` (DLL path on PATH, GPU),
`permute_and_run.py` reports **all 6 permutations run fine and give identical
outputs** (`compare_ep.py` confirms the OpenVINO EP really ran on GPU, not a CPU
fallback). The input-binding bug does not reproduce on this official-release
stack. Note both ORT and OpenVINO versions differ from the dev build below, so the
regression cannot be pinned to either component alone.

### 4. On KataGo's self-built dev ORT 1.29.0 + OpenVINO 2026.2 stack, the failure depends on the branch

Reproduced by temporarily gating the workaround with an env var
(`KATAGO_DEBUG_MASTER_INPUT_ORDER=1` forces the master order; one-line change in
`cpp/neuralnet/onnxbackend.cpp`, rebuilt, then reverted):

| declaration order | `requireExactNNLen=true` | `requireExactNNLen=false` |
|---|---|---|
| master `(Mask, Spatial, Global)` | **crashes** — `can't handle input tensor with name: parameter:InputSpatial, because model input (shape=[?,22,19,19]) and tensor (shape=[1,1,19,19]) are incompatible` (exit 0xC0000409) | runs fine (98.8 visits/s), output correct |
| workaround `(Spatial, Global, Mask)` | runs fine | runs fine |

`compare_two_models.py` confirms the `requireExactNNLen=false` master-order model
and the workaround-order model are semantically identical (maxdiff = 0), so the
master order working there is *correct*, not "runs but wrong".

### Conclusions

- The workaround is **necessary** for the `requireExactNNLen=true` branch on
  ORT 1.29 / OpenVINO 2026.2 (this reproduces the exact error the PR comment
  documents), and it also happens to be harmless on the `=false` branch.
- It is **fragile in exactly the way the review warned**: the EP's accepted input
  order is not the graph's first-reference order, is not derivable from the code,
  and *differs between branches* (master order crashes only on `=true`).
- The failure reproduces on the self-built dev ORT (1.29.0) + OpenVINO 2026.2
  stack but **not** on the official release line (ORT 1.24.1 + OpenVINO 2025.4),
  where all 6 orders work. The two stacks differ in *both* ORT and OpenVINO
  versions, so the regression cannot be attributed to either version alone.
  KataGo feeds inputs by name (ORT Run API), so the mismatch is inside the EP —
  see the source-level root cause below.

## Root cause (ORT source)

The ONNX Runtime KataGo links is a **self-built dev build** (`onnxruntime` main
branch, HEAD at local tag `test-tag-7528`, version string 1.29.0) — not an
official release.

### Where the mis-indexing comes from

The regression is a 3-line change in the input-index loop of `backend_manager.cc`
(executed when a subgraph is handed to the OpenVINO EP), introduced by commit
`dfc27cd7c7e` "[OVEP] OpenVINO EP Features Release 1.23 (#25262)"
(2025-07-04):

```cpp
for (uint32_t index = 0; const auto& node : subgraph.GetInputs()) {
  if (subgraph.GetGraph().GetConsumerNodes(node->Name()).size() == 0) {
    continue;  // Skip if the input is a dangling node   <-- added by dfc27cd7c7e
  }
  subgraph_context_.input_names.insert({node->Name(), index++});
}
```

`index` is documented as the position of a graph input **among the fused node's
`inputDefs`** (which also contains initializers). Skipping a dangling input
without skipping its `inputDefs` position desyncs `index` from the position that
`context.GetInput(index)` later uses. At inference time (`basic_backend.cc:363`,
static-shape path) the data is fetched **by ORT index** but bound **by name**:

```cpp
infer_request->SetTensor(input_info.name, input_info.type, input_info.shape,
                         context.GetInput(input_info.onnx_index).GetTensorRawData());
```

When a dead input sits ahead of a live one in declaration order — e.g. the dead
`InputMask` in `requireExactNNLen=true` — the live input gets a too-small index
and the `[1,1,19,19]` mask tensor is routed into the `InputSpatial` port: the
exact error reproduced in step 5. This also explains the workaround: declaring
the dead `InputMask` last keeps the surviving indices contiguous.

> **Confidence note:** this mechanism is an interpretation that matches every
> observation (real-stack crash, `=false` branch working, workaround working),
> but the "fused node `inputDefs`" semantics of `context.GetInput` has not been
> verified directly (e.g. by reverting the 3 lines and re-testing).

### When it was introduced

| onnxruntime version | dangling-skip code present |
|---|---|
| v1.22.0 / v1.22.1 / v1.22.2 | no |
| **v1.23.0** (first) … v1.24.x, v1.25.x, v1.26, v1.27, v1.28.0 | **yes** |
| KataGo's self-built dev 1.29.0 | yes |

The code exists in official ONNX Runtime from **v1.23.0** onwards (confirmed by
`git show <tag>`). Note **"code present" ≠ "bug reproduces"**: whether it
actually crashes also depends on how ORT/OpenVINO treat the dead input, which is
why the official `onnxruntime-openvino 1.24.1` + OpenVINO 2025.4 stack (step 4)
does not reproduce it. That difference is not yet explained and is out of scope
for this repo.

KataGo feeds inputs by name through the ORT Run API (the conforming interface);
the mismatch is purely inside the EP's binding logic.

## Environment used

- KataGo real stack: ONNX Runtime **1.29.0** + OpenVINO **2026.2** (built from
  source, Intel Arc B580 GPU). Used for dumping models and for the real-stack
  reproduction (step 5).
- Python (system): `onnx`, `onnxruntime` (CPU EP). Used for steps 2–3 and 5.
- Python OpenVINO EP: `onnxruntime-openvino 1.24.1` + OpenVINO **2025.4.1**
  (version-matched; the bundled EP needs the matching `openvino.dll` on PATH).
  Used for step 4.

## Reproduction

### 0. Prerequisites
- A KataGo ONNX build (`onnxProvider = openvino`) and a `.bin.gz` model.
- Python with `onnx` + `onnxruntime` for CPU/static steps.
- For the OpenVINO EP Python step: `pip install onnxruntime-openvino==1.24.1`
  and a matching `openvino` 2025.4.x wheel extracted to a temp dir so its
  `openvino/libs` DLL folder can be put on `PATH` *without* disturbing any newer
  OpenVINO your environment already has.

### 1. Dump the two branch variants (355 MB each, not committed here)
```bat
set PATH=D:\path\to\release_pkg;%PATH%
set KATAGO_DUMP_ONNX=D:\dump\rtlen_true.onnx
katago benchmark -config gpu_optimized.cfg -model <model.bin.gz> -threads 4 -visits 5
rem requireExactNNLen=true (default benchmark behavior)

set KATAGO_DUMP_ONNX=D:\dump\rtlen_false.onnx
katago benchmark -config gpu_optimized_rtlen_false.cfg -model <model.bin.gz> -threads 4 -visits 5
rem requireMaxBoardSize=false -> requireExactNNLen=false branch
```

### 2. Static first-consumption analysis
```bash
python analyze_inputs.py rtlen_true.onnx rtlen_false.onnx
```

### 3. Name-bound (CPU) invariance across all 6 orders
```bash
python permute_and_run.py rtlen_false.onnx CPUExecutionProvider
python permute_and_run.py rtlen_true.onnx CPUExecutionProvider
```

### 4. OpenVINO EP behavior (2025.4 stack)
```bash
set PATH=<temp-openvino-2025.4>\openvino\libs;%PATH%
python compare_ep.py rtlen_false.onnx        # confirms EP actually runs on GPU
python permute_and_run.py rtlen_false.onnx openvino:GPU
```

### 5. Real-stack reproduction of the master-order failure (ORT 1.29 + OpenVINO 2026.2)
Temporarily change one line in `cpp/neuralnet/onnxbackend.cpp`:
```cpp
ctx->providerName == "openvino" && getenv("KATAGO_DEBUG_MASTER_INPUT_ORDER") == nullptr
```
rebuild, then:
```bat
set KATAGO_DEBUG_MASTER_INPUT_ORDER=1
katago benchmark -config gpu_optimized.cfg -model <model.bin.gz> -threads 4 -visits 20
rem -> crashes with the InputSpatial shape mismatch (requireExactNNLen=true)
katago benchmark -config gpu_optimized_rtlen_false.cfg -model <model.bin.gz> -threads 4 -visits 20
rem -> runs fine (requireExactNNLen=false)
```
Revert the one-liner before committing.

## Files

| file | purpose |
|---|---|
| `analyze_inputs.py` | declared vs first-consumption order per model |
| `permute_and_run.py` | all 6 input-order permutations x one provider; reports identical/broken |
| `compare_ep.py` | proves the OpenVINO EP runs on GPU (not a silent CPU fallback) |
| `compare_two_models.py` | proves two declaration orders are semantically identical |
| `gpu_optimized_rtlen_false.cfg` | config to dump/run the `requireExactNNLen=false` branch |

## Constraints

- The dumped ONNX models (~355 MB each) are **not** committed; they must be
  regenerated with `KATAGO_DUMP_ONNX` (step 1).
- The 2025.4 OpenVINO EP result is from `onnxruntime-openvino 1.24.1`; the real
  KataGo stack (1.29.0 + 2026.2) is only exercised through the binary in step 5.
- Inputs in the scripts use a full-board all-ones mask; random mask values would
  overflow the attention mask bias and produce NaN regardless of ordering.
