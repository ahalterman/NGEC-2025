"""How many CPU threads should the PyTorch encoders use?

PyTorch defaults to one thread per core. That is not obviously right when
`llama-server` is also on the box, so this times the encoders that are actually
hot and lets you pick a number rather than guess one.

Two are measured:

  * the **event classifier's** encoder (step 1), read from
    `event_models_v2/metadata.json` so this tracks whatever the models were
    trained with. On a CPU host this is usually the largest PyTorch cost in a
    run.
  * the **agent matcher's** encoder (step 5), `AGENT_ENCODER` in
    `ngec/actors/common.py`.

The Wikipedia matcher's own encoder is deliberately not measured. It defaults to
`static-retrieval-mrl-en-v1`, which is a lookup table with no transformer in it,
so thread count does not apply. See `WIKI_ENCODERS`.

Run once per thread count and compare -- the answer is hardware-specific:

    for t in 1 2 4 6 8; do python torch_threads.py $t; done
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Thread counts must be set before torch is imported to take effect everywhere.
NT = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(NT)
os.environ["MKL_NUM_THREADS"] = str(NT)

import torch                                          # noqa: E402
torch.set_num_threads(NT)

sys.path.insert(0, REPO)
from sentence_transformers import SentenceTransformer  # noqa: E402
from ngec.actors.common import AGENT_ENCODER           # noqa: E402

# A document of roughly the length the classifier chunks at.
DOC = ("Thousands of protesters gathered in Paris on Tuesday to demonstrate "
       "against the government's proposed pension reforms, which would raise "
       "the retirement age by two years. President Emmanuel Macron said the "
       "reforms would proceed despite the demonstrations. Police used tear gas "
       "to disperse a crowd that had blocked traffic near the Place de la "
       "Republique. Union leaders said they would call further strikes if the "
       "government did not withdraw the bill. ") * 3

MENTIONS = ["the interior ministry", "Emmanuel Macron", "riot police",
            "Union leaders", "the government"] * 10


def bench(name, model, texts, reps=3):
    model.encode(texts[:2], show_progress_bar=False)      # warm
    ts = []
    for _ in range(reps):
        t0 = time.time()
        model.encode(texts, show_progress_bar=False)
        ts.append(time.time() - t0)
    print(f"{name}\t{NT}\t{min(ts):.3f}\t{sum(ts) / len(ts):.3f}")


print("model\tthreads\tbest_s\tmean_s")

meta_path = os.path.join(REPO, "ngec", "assets", "event_models_v2",
                         "metadata.json")
encoder_name = json.load(open(meta_path))["encoder"]
bench("classifier_enc", SentenceTransformer(f"sentence-transformers/{encoder_name}"),
      [DOC] * 8)

bench("agent_enc", SentenceTransformer(AGENT_ENCODER), MENTIONS)
