#!/usr/bin/env python
"""
explain_v4_cudnn.py: rerun explain_v4.py for the M7 Transformer with cuDNN kept ON
during the static-covariate SHAP step.

Why: for K, SHAP failed with --shap-batch 5 when cuDNN was disabled:
  base pipeline M7_explainability.run_shap_tabular wraps the explainer in
      with torch.backends.cudnn.flags(enabled=False):
  That workaround exists for cuDNN-RNN backward in the M7_LSTM variant only. With
  cuDNN off, convolutions use PyTorch's native kernel, which cannot split tensors
  above 2^31 elements and raises "canUse32BitIndexMath". The 10 m K tiles in a long
  image sequence exceed that limit even for a single SHAP batch. With cuDNN on,
  PyTorch splits the batch to 32-bit-safe chunks. The SHAP estimator, background,
  seed and inputs are unchanged.

Safe only for --temporal transformer (the M7 used in the paper); do NOT use for M7_LSTM.
Nothing in explain_v4.py or base pipeline is edited; the patches are in-process only.

Usage (run from final_runs/, next to explain_v4.py):
    python explain_v4_cudnn.py --nutrient K --seqfix --sampEnd [--shap-batch 5]
"""
import contextlib
import functools
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_RUNS = HERE if os.path.exists(os.path.join(HERE, "explain_v4.py")) else os.path.dirname(HERE)
if not os.path.exists(os.path.join(FINAL_RUNS, "explain_v4.py")):
    sys.exit(f"explain_v4.py not found in {HERE} or its parent")

batch = 5
if "--shap-batch" in sys.argv:
    i = sys.argv.index("--shap-batch")
    batch = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]

import torch
import shap


@contextlib.contextmanager
def _cudnn_flags_noop(*args, **kwargs):
    # keep the current cuDNN settings (enabled) instead of switching cuDNN off
    yield


torch.backends.cudnn.flags = _cudnn_flags_noop
shap.GradientExplainer = functools.partial(shap.GradientExplainer, batch_size=batch)
print(f"[cudnn-on] cuDNN kept enabled for SHAP; GradientExplainer batch_size={batch}", flush=True)

os.chdir(FINAL_RUNS)
sys.path.insert(0, FINAL_RUNS)
script = os.path.join(FINAL_RUNS, "explain_v4.py")
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
