#!/usr/bin/env python3
# Confirm the OpenVINO EP is actually used (not silently falling back to CPU):
# run the same model + inputs on OpenVINO:GPU and on CPU and compare outputs.
# GPU (FP16) must differ slightly from CPU (FP32).
#
# Requires onnxruntime-openvino and a matching OpenVINO DLL on PATH (see README).
#
# Usage: python compare_ep.py <model.onnx>
import sys
import numpy as np
import onnx
import onnxruntime as ort
from permute_and_run import name_to_shape


def run(model_path, providers):
    with open(model_path, "rb") as f:
        m = onnx.load_model_from_string(f.read())
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=providers)
    n2s = name_to_shape(m)
    rng = np.random.default_rng(0)
    feeds = {}
    for n in sorted(n2s):
        if n == "InputMask":
            feeds[n] = np.ones(n2s[n], np.float32)
        else:
            feeds[n] = rng.uniform(0, 1, n2s[n]).astype(np.float32)
    outs = sess.run(None, feeds)
    print("active providers:", sess.get_providers())
    return outs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python compare_ep.py <model.onnx>")
        sys.exit(1)
    ov = run(sys.argv[1], [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"])
    print("--- now CPU ---")
    cpu = run(sys.argv[1], ["CPUExecutionProvider"])
    for i, (a, b) in enumerate(zip(ov, cpu)):
        diff = np.abs(a - b)
        print(f"out[{i}] shape={a.shape} ov_maxdiff_vs_cpu={diff.max():.6g} "
              f"allclose={np.allclose(a, b, rtol=1e-2, atol=1e-3)}")
