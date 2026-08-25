# SeMoCo-Generator

[Model](https://huggingface.co/poisonousID/SeMoCo) | [Tokenizer](https://github.com/OMEGA-i/SeMoCo-Tokenizer)

SeMoCo introduces a semantic-first motion codec that organizes discrete
motion tokens by semantic roles, disentangling high-level motion states from
fine-grained kinematic details to improve autoregressive text-to-motion
generation.

## Quick Start

### 1. Environment

Requires Python 3.12 and PyTorch (CUDA).

```bash
git clone --recursive https://github.com/OMEGA-i/SeMoCo-Generator.git
cd SeMoCo-Generator
git submodule update --init --recursive   # if you cloned without --recursive
uv sync                                   # or: pip install -e .
```

Anything that turns codes back into motion — generation, evaluation, the
prediction benchmark — also needs the `decode` extra, which brings in the
SOMA-X forward-kinematics runtime. Other extras: `tmr` for the retrieval
evaluator, `render` for MP4/GIF animations, `viewer` for the interactive 3D
player, `dev` for tests. Example: `uv sync --extra decode --extra render`.
`Dockerfile` builds a CUDA image with the tokenizer and these extras in place.

### 2. The frozen tokenizer

Generation and evaluation decode motion codes through the SeMoCo tokenizer;
training does not need it.

```bash
git lfs install   # the tokenizer keeps its SOMA-X assets in LFS
git clone --recursive https://github.com/OMEGA-i/SeMoCo-Tokenizer.git
export SOMA_TOKENIZER_ROOT=/path/to/SeMoCo-Tokenizer
```

### 3. Pretrained weights

The generators and the tokenizer they decode through share one
[Hugging Face repository](https://huggingface.co/poisonousID/SeMoCo):

```bash
hf download poisonousID/SeMoCo --include 'generator/*' 'tokenizer/split_branch_sem.pt' \
    --local-dir checkpoints/
export SOMA_TOKENIZER_CHECKPOINT=checkpoints/tokenizer/split_branch_sem.pt
```

| File | Model | Params |
|---|---|---|
| `generator/lite.pt` | Lite, text-to-motion | 188M |
| `generator/base.pt` | Base, text-to-motion | 391M |
| `generator/prior_lite.pt` | Lite, unconditional motion prior | 199M |
| `generator/prior_base.pt` | Base, unconditional motion prior | 419M |
| `tokenizer/split_branch_sem.pt` | SeMoCo tokenizer | 22.9M |

The two motion priors are only needed for the prediction benchmark; the
text-to-motion models do not depend on them.

### 4. External assets

`third_party/kimodo` (a git submodule) provides the SOMA skeleton definitions.
Everything else is fetched on demand:

```bash
python -m semoco_generator.tools.fetch_assets fetch
python -m semoco_generator.tools.fetch_assets verify
```

| Asset | Purpose | Where |
|---|---|---|
| kimodo | `SOMASkeleton77` definitions and the retrieval-teacher backbone | `third_party/kimodo` submodule; `SKIP_MOTION_CORRECTION_IN_SETUP=1 uv pip install -e third_party/kimodo` |
| TMR-SOMA-RP-v1 | SOMA/TMR retrieval evaluator | fetched from [Hugging Face](https://huggingface.co/nvidia/TMR-SOMA-RP-v1) |
| LLM2Vec | R-precision text encoder for the SOMA/TMR track | fetched automatically, but its base [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) is gated — without access, run that track with `--no-rprecision` |
| HumanML3D + evaluator | HumanML3D track (`text_mot_match`, GloVe vocabulary) | fetched automatically |
| SMPL / SMPL-X | mesh decode for cross-representation conversion | register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/); set `$SMPL_MODEL_PATH` / `$SMPLX_MODEL_PATH` |

`SKIP_MOTION_CORRECTION_IN_SETUP=1` skips kimodo's bundled C++ extension,
which needs cmake and is not used by this repo.

### 5. Generate motion from text

```bash
python -m semoco_generator.eval.t2m_infer \
    --checkpoint checkpoints/generator/base.pt \
    --prompts "a person walks forward" "a person jumps in place" \
    --out-dir runs/infer/base --render mp4
```

`viewer/viser_live_viewer.py` streams the skeleton as it is generated.

## Training

Artifacts are addressed with `local://…` URIs resolved against a data root:

```bash
export MOTIONVERSE_DATA_ROOT=/path/to/your/data
```

Training reads UMR-499 features from the parquet shards produced by the
tokenizer's dataset build. Export paired motion codes and text embeddings with
the frozen tokenizer, then train:

```bash
python -m semoco_generator.tools.export_t2m_dataset \
    --parquet-dir <release>/derived_umr_<hash> --split train \
    --out-dir local://t2m_codes --checkpoint <tokenizer>/best.pt \
    --text-encoder flan

torchrun --nproc_per_node=4 -m semoco_generator.train.train_t2m \
    --config configs/t2m_150m_flan.yaml
```

The released weights come from `configs/t2m_150m_flan.yaml` and
`configs/t2m_400m_flan.yaml`. `--text-encoder` must name the same encoder as
the config you train, since the export step bakes those embeddings into the
code store.

## Evaluation

Two tracks run the full generation protocol, on a downloaded checkpoint or on
your own run:

```bash
# HumanML3D — official text_mot_match evaluator
python -m semoco_generator.eval.cli run --track smpl_hml --mode main \
    --semoco-checkpoint checkpoints/generator/lite.pt \
    --semoco-tokenizer checkpoints/tokenizer/split_branch_sem.pt

# SOMA/TMR — TMR-SOMA-RP-v1 retrieval evaluator on a SOMA code store
python -m semoco_generator.eval.cli run --track soma_tmr --mode main \
    --codes-root local://t2m_codes --split test \
    --semoco-checkpoint checkpoints/generator/lite.pt \
    --semoco-tokenizer checkpoints/tokenizer/split_branch_sem.pt
```

Asset paths default to what `fetch_assets` installs; `--data-root` and
`--glove-root` override them. The HumanML3D track can seed forward kinematics
with per-clip body anchors, otherwise it uses a canonical body:

```bash
python -m semoco_generator.tools.precompute_hml_anchors --data-root /path/to/HumanML3D
```

Teacher-forced token metrics over an exported code store:

```bash
python -m semoco_generator.eval.t2m_token_eval \
    --checkpoint checkpoints/generator/lite.pt \
    --codes-root local://t2m_codes --split test
```

### Motion prediction

Unconditional prediction, reported as ADE/FDE in metres. This observes the
first 20% of a clip's tokens and generates the rest, so it runs on a motion
prior (`prior_lite.pt`, or your own `configs/motion_gpt_*.yaml` run) rather
than a text-to-motion checkpoint. Ground truth is the forward kinematics of the
real UMR-499 features, which never pass through the codebook:

```bash
python -m semoco_generator.eval.prediction.build_gt \
    --parquet-dir <release>/derived_umr_<hash> --split test \
    --out-dir local://pred_gt/test

python -m semoco_generator.eval.prediction.predict \
    --checkpoint checkpoints/generator/prior_lite.pt \
    --tokenizer-checkpoint checkpoints/tokenizer/split_branch_sem.pt \
    --codes-root local://t2m_codes --parquet-dir <release>/derived_umr_<hash> \
    --split test --out-dir local://pred_out/test --max-tokens 31

python -m semoco_generator.eval.prediction.score \
    --pred-dir local://pred_out/test --gt-dir local://pred_gt/test \
    --out-json runs/prediction/metrics.json
```

`--max-tokens 31` caps the horizon at 2s; drop it to score whole clips. Adding
`--num-samples K` draws K rollouts per clip and keeps the best against the
ground truth, which `score` then reports as min-ADE (`_pred.npy`) and min-FDE
(`--pred-suffix _predfde.npy`).

## Acknowledgements

This project builds on [SeMoCo](https://github.com/OMEGA-i/SeMoCo-Tokenizer)
for motion tokenization, NVIDIA's
[TMR-SOMA-RP-v1](https://huggingface.co/nvidia/TMR-SOMA-RP-v1) retrieval
evaluator via the [kimodo](https://github.com/nv-tlabs/kimodo) skeleton
library, and the official [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
evaluation protocol. These dependencies remain under their own licenses.

## License

Apache-2.0, see [`LICENSE`](LICENSE). External dependencies retain their own
licenses.
