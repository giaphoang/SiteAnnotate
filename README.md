<h1 align="center">SiteAnnotate</h1>
**SiteAnnotate** integrates annotation automation solutions to create training data for robotics, autonomy, computer vision, and Vision-Language-Action (VLA) systems.

It reduces the manual work needed to turn off-road camera–LiDAR recordings into consistent, reviewed training datasets.

| Item | Description |
| --- | --- |
| Input | RELLIS-3D camera sequences, LiDAR point clouds, timestamps, sensor calibration, and reference semantic labels reserved for evaluation |
| Processing | Camera–LiDAR alignment → Cosmos Curator filtering/deduplication → Cosmos Reason captioning → annotation review → quality validation |
| Output | Curated sensor samples, reviewed scene captions, approved annotations, versioned dataset manifests, and quality reports |

## Start here

Run the offline end-to-end demo. It creates synthetic RELLIS data in a
temporary directory and needs no GPU, model weights, or dataset download:

```bash
make demo
```

## Run a real RELLIS-3D sequence

Install Python 3.10+, [`uv`](https://docs.astral.sh/uv/), and an extracted
RELLIS-3D tree. Put the tree at `data/RELLIS-3D`, then run:

```bash
make bootstrap-sources
uv sync --extra runtime --extra dev

make ingest RELLIS_ROOT=data/RELLIS-3D SEQUENCE=00000 RUN=runs/rellis-00000
make predict-semantics RUN=runs/rellis-00000
make review RUN=runs/rellis-00000
```

Without weights, `predict-semantics` stages reference labels for evaluation;
use reviewed model predictions for training annotations.

Open the 3D box reviewer and semantic-label editor in separate terminals:

```bash
make box-review RUN=runs/rellis-00000
PYTHONPATH=src python3 -m siteannotate semantic-editor \
  --review-root runs/rellis-00000/review/sustech
```

Frame timestamps are preserved in `frames.jsonl` and
`curation/frame_links.jsonl`, keeping every curated camera sample linked to
its LiDAR scan. Review edits are copied into the run, so downloaded RELLIS
files are not modified.

## Optional AI stages

PAIDF curation, Cosmos Reason captioning, and SalsaNext predictions require
their upstream runtimes and model weights. Once those services are configured:

```bash
make curate RUN=runs/rellis-00000 MODELS_DIR=/models/cosmos-curator \
  FFMPEG_DIR=/opt/cosmos-curator-ffmpeg

export COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL=http://127.0.0.1:8000/v1
export COSMOS_CURATOR_OPENAI_CAPTION_API_KEY=EMPTY
export COSMOS_REASON_MODEL=nvidia/Cosmos-Reason1-7B
PYTHONPATH=src python3 -m siteannotate caption --run runs/rellis-00000

make predict-semantics RUN=runs/rellis-00000 WEIGHTS_DIR=/models/salsanext
```

After approving the review gates, validate and publish:

```bash
make validate RUN=runs/rellis-00000
make publish RUN=runs/rellis-00000
```

The integration is built from [PAIDF Curation and Retrieval](https://github.com/NVIDIA/paidf-curation-and-retrieval),
[Cosmos Reason1](https://github.com/nvidia-cosmos/cosmos-reason1),
[SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS), and
[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D).
