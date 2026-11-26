#!/usr/bin/env python
"""
explain_v4_smallbatch.py: rerun explain_v4.py with a smaller SHAP batch.

Why: in the K run, SHAP failed for 12 of 50 samples with
"canUse32BitIndexMath": shap.GradientExplainer evaluates batch_size=50 copies of the
sample's fixed image sequence at once, and the 10 m K tiles x long sequences exceed the
2^31-element limit of a single conv call. The base pipeline retry only catches "out of memory".
batch_size only sets how many expected-gradient draws go through the model per call;
the estimate (nsamples draws over the same background) is unchanged.

Usage (same args as explain_v4.py):
    python explain_v4_smallbatch.py --nutrient K --seqfix --sampEnd [--shap-batch 5]
Nothing in explain_v4.py or base pipeline is edited; the patch is in-process only.
"""
import functools
import runpy
import sys

batch = 5
if "--shap-batch" in sys.argv:
    i = sys.argv.index("--shap-batch")
    batch = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]

import shap
_GE = shap.GradientExplainer
shap.GradientExplainer = functools.partial(_GE, batch_size=batch)
print(f"[smallbatch] shap.GradientExplainer batch_size={batch}", flush=True)

sys.argv[0] = "explain_v4.py"
runpy.run_path("explain_v4.py", run_name="__main__")
