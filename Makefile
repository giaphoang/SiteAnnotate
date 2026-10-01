.PHONY: help bootstrap-sources install demo test lint ingest curate caption predict-semantics review box-review validate publish retrieve

PYTHON ?= python3
RUN ?= runs/rellis-demo
RELLIS_ROOT ?= data/RELLIS-3D
SEQUENCE ?= 00000
MODELS_DIR ?= models
FFMPEG_DIR ?= $(CURDIR)/.cache/cosmos-curator-ffmpeg
CONFIG ?=
WEIGHTS_DIR ?=
PREDICTIONS_DIR ?=

help:
	@echo "SiteAnnotate targets:"
	@echo "  bootstrap-sources  clone and pin the four approved GitHub sources"
	@echo "  install            install the local package with uv"
	@echo "  demo               run the offline end-to-end demo"
	@echo "  test               run offline tests"
	@echo "  ingest             normalize an extracted RELLIS sequence"
	@echo "  curate             run PAIDF split/embedding stages"
	@echo "  caption            call the Cosmos Reason OpenAI-compatible endpoint"
	@echo "  predict-semantics  run or stage SalsaNext semantic predictions"
	@echo "  review             stage SUSTechPOINTS data and semantic review files"
	@echo "  box-review         launch SUSTechPOINTS camera-LiDAR box review"
	@echo "  validate           enforce publication gates"
	@echo "  publish            create the validated dataset package"
	@echo "  retrieve           run PAIDF Data Mining over curated embeddings"

bootstrap-sources:
	bash scripts/bootstrap_sources.sh

demo: test

install:
	uv sync --extra runtime --extra dev

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m compileall -q src tests
	@if command -v ruff >/dev/null 2>&1; then ruff check src tests; fi

ingest:
	PYTHONPATH=src $(PYTHON) -m siteannotate ingest-extracted --root $(RELLIS_ROOT) --sequence $(SEQUENCE) --run $(RUN)

curate:
	PYTHONPATH=src $(PYTHON) -m siteannotate curate --run $(RUN) --models-dir $(MODELS_DIR) --ffmpeg-dir $(FFMPEG_DIR) $(if $(CONFIG),--config $(CONFIG),)

caption:
	PYTHONPATH=src $(PYTHON) -m siteannotate caption --run $(RUN)

predict-semantics:
	PYTHONPATH=src $(PYTHON) -m siteannotate predict-semantics --run $(RUN) $(if $(WEIGHTS_DIR),--weights-dir $(WEIGHTS_DIR),)

review:
	PYTHONPATH=src $(PYTHON) -m siteannotate stage-review --run $(RUN) $(if $(PREDICTIONS_DIR),--predictions-dir $(PREDICTIONS_DIR),)

box-review:
	PYTHONPATH=src $(PYTHON) -m siteannotate box-review --run $(RUN)

validate:
	PYTHONPATH=src $(PYTHON) -m siteannotate validate --run $(RUN)

publish:
	PYTHONPATH=src $(PYTHON) -m siteannotate publish --run $(RUN)

retrieve:
	PYTHONPATH=src $(PYTHON) -m siteannotate retrieve --run $(RUN)
