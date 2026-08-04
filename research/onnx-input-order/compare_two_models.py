#!/usr/bin/env python3
# Compare the outputs of two ONNX models (same graph, different input declaration
# order) under identical inputs, on the CPU EP. A maxdiff of 0 confirms that
# reordering inputs is semantically a no-op for a name-bound backend.
#
# Usage: python compare_two_models.py <modelA.onnx> <modelB.onnx>
import sys
import numpy as np
import onnx
import onnxruntime as ort
from permute_and_run import name_to_shape


def run(path):
    m = onnx.load(path)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    n2s = name_to_shape(m)
    rng = np.random.default_rng(0)
    fixed = {}
    for n in sorted(n2s):
        if n == "InputMask":
            fixed[n] = np.ones(n2s[n], np.float32)
        else:
            fixed[n] = rng.uniform(0.0, 1.0, n2s[n]).astype(np.float32)
    feeds = {vi.name: fixed[vi.name] for vi in m.graph.input}
    return sess.run(None, feeds)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python compare_two_models.py <modelA.onnx> <modelB.onnx>")
        sys.exit(1)
    a = run(sys.argv[1])
    b = run(sys.argv[2])
    print(f"comparing:\n  A={sys.argv[1]}\n  B={sys.argv[2]}")
    for i, (x, y) in enumerate(zip(a, b)):
        print(f"  out[{i}] shape={x.shape} allclose={np.allclose(x, y, rtol=1e-5, atol=1e-7)} "
              f"maxdiff={np.abs(x - y).max():.3g}")
