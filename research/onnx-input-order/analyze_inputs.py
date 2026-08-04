#!/usr/bin/env python3
# Static analysis: declared input order vs. first-consumption order.
#
# For each dumped ONNX model, find the order in which each graph input is FIRST
# referenced by a node, following node creation order (KataGo builds nodes in
# topological order). This is the "consumption order" that the review questioned:
# the OpenVINO EP fix (alignInputsToConsumptionOrder) claims inputs must be declared
# in consumption order, but the actual first-reference order depends on the graph
# build path (requireExactNNLen, HumanSL, ...).
#
# Usage: python analyze_inputs.py <model.onnx> [model2.onnx ...]
import sys
import onnx


def analyze(path):
    m = onnx.load(path)
    g = m.graph
    inputs = [i.name for i in g.input]
    init_names = {init.name for init in g.initializer}

    print(f"===== {path} =====")
    print(f"declared input order: {inputs}")
    print(f"input count: {len(g.input)}  node count: {len(g.node)}  init count: {len(g.initializer)}")

    # Producer map: tensor name -> node index (validates that node order is topo order).
    producer = {}
    for idx, n in enumerate(g.node):
        for o in n.output:
            producer[o] = idx

    input_set = set(inputs)
    first_use = {}
    first_use_order = []
    for idx, n in enumerate(g.node):
        for inp in n.input:
            if not inp:
                continue
            if inp in input_set and inp not in first_use:
                first_use[inp] = (idx, n.op_type)
                first_use_order.append((inp, idx, n.op_type))

    print("first-consumption order (input name, node idx, op):")
    for name, idx, op in first_use_order:
        print(f"  {name:14s} first used at node[{idx}] op={op}")
    for name in inputs:
        if name not in first_use:
            print(f"  {name:14s} NEVER consumed (dead input)")
    if first_use_order:
        consumption = [t[0] for t in first_use_order]
        print(f"consumption order == declared order? {consumption == inputs}")
        print(f"  consumption: {consumption}")
        print(f"  declared:    {inputs}")
    print()


for p in sys.argv[1:]:
    analyze(p)
