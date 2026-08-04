#!/usr/bin/env python3
# For a dumped ONNX model, test ALL 6 permutations of input declaration order.
# Each permutation is re-serialized in-memory and run once through ORT with a given
# provider. Records success/failure and, across permutations, whether outputs are
# identical (a name-bound backend must be insensitive to declaration order).
#
# Usage:
#   python permute_and_run.py <model.onnx> CPUExecutionProvider
#   python permute_and_run.py <model.onnx> openvino:GPU
#       (openvino:<device> -> OpenVINOExecutionProvider with device_type=<device>)
import sys
import itertools
import numpy as np
import onnx
import onnxruntime as ort


def permute_model(src_bytes, perm):
    m = onnx.load_model_from_string(src_bytes)
    ins = list(m.graph.input)
    del m.graph.input[:]
    for p in perm:
        m.graph.input.append(ins[p])
    return m


def name_to_shape(m):
    out = {}
    for vi in m.graph.input:
        t = vi.type.tensor_type
        shape = []
        for d in t.shape.dim:
            dv, dp = d.dim_value, d.dim_param
            shape.append(dv if dv > 0 else 1)
        out[vi.name] = shape
    return out


def make_feeds(m):
    # Deterministic per NAME (sorted order), so feeds do not depend on permutation.
    # InputMask must be a real 0/1 board mask (all-ones = full board); random values
    # blow up the attention mask bias (mask-1)*(-3e4) -> softmax overflow.
    n2s = name_to_shape(m)
    rng = np.random.default_rng(0)
    fixed = {}
    for n in sorted(n2s):
        if n == "InputMask":
            fixed[n] = np.ones(n2s[n], np.float32)
        else:
            fixed[n] = rng.uniform(0.0, 1.0, n2s[n]).astype(np.float32)
    return {vi.name: fixed[vi.name] for vi in m.graph.input}


def build_providers(provider_spec):
    if ":" in provider_spec:
        pname, device = provider_spec.split(":", 1)
        if pname == "openvino":
            pname = "OpenVINOExecutionProvider"
        return [(pname, {"device_type": device}), "CPUExecutionProvider"]
    pname = "OpenVINOExecutionProvider" if provider_spec == "openvino" else provider_spec
    if pname == "OpenVINOExecutionProvider":
        return [pname, "CPUExecutionProvider"]
    return [pname]


def run_once(src_bytes, perm, providers):
    m = permute_model(src_bytes, perm)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=providers)
        out = sess.run(None, make_feeds(m))
        ok = all(np.isfinite(o).all() for o in out)
        if not ok:
            bad = [float(np.isfinite(o).mean()) for o in out]
            return (None, f"nonfinite_ratio={bad}")
        return (out, None)
    except Exception as e:
        lines = str(e).splitlines()
        return (None, "FAIL: " + " | ".join(l for l in lines[:2] if l.strip()))


def main():
    if len(sys.argv) < 3:
        print("usage: python permute_and_run.py <model.onnx> <provider>")
        print("  provider: CPUExecutionProvider | openvino:GPU | OpenVINOExecutionProvider[:device]")
        sys.exit(1)
    src_path, provider_spec = sys.argv[1], sys.argv[2]
    with open(src_path, "rb") as f:
        src_bytes = f.read()
    providers = build_providers(provider_spec)
    names = [i.name for i in onnx.load_model_from_string(src_bytes).graph.input]
    print(f"===== {src_path} | provider={provider_spec} =====", flush=True)
    print(f"base declared order: {names}", flush=True)
    base_out = None
    for perm in itertools.permutations(range(3)):
        decl = [names[p] for p in perm]
        out, err = run_once(src_bytes, perm, providers)
        if err:
            print(f"  perm={perm} declared={decl}: {err}", flush=True)
            continue
        if base_out is None:
            base_out = out
            print(f"  perm={perm} declared={decl}: OK (baseline)", flush=True)
        else:
            equal = all(np.array_equal(a, b) for a, b in zip(base_out, out))
            close = all(np.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(base_out, out))
            print(f"  perm={perm} declared={decl}: OK  identical_to_baseline={equal} allclose={close}", flush=True)


if __name__ == "__main__":
    main()
