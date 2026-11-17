# MLServer Benchmarks — single entry point
#
# Everything is a make target. Start here:
#   make setup        bootstrap the cluster environment (once)
#   make smoke-test   one gated trial to prove the setup
#   make run-all      every domain (hours)
#
# Per-plan runs:
#   make run-plan DOMAIN=wire-format PLAN=stock-vs-binary
#
# Environment overrides (also settable as BENCHMARK_* env vars):
#   NAMESPACE=mlserver-perf  CONTEXT=  CLIENT_POD=  IMAGE=

PYTHON  ?= python3
DOMAIN  ?=
PLAN    ?=
NAMESPACE ?= mlserver-perf
ENGINE   ?= local
REPO_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

RUN = $(PYTHON) tools/run.py --namespace $(NAMESPACE)

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup ---

.PHONY: setup
setup: ## Full bootstrap: namespace -> client pod -> models -> references -> smoke test
	bash infrastructure/setup.sh

.PHONY: verify
verify: ## Verify the environment is ready to benchmark
	bash infrastructure/verify.sh

.PHONY: teardown
teardown: ## Scale everything down (keeps namespace and PVC data)
	bash infrastructure/teardown.sh

.PHONY: ceiling-probe
ceiling-probe: ## F-02 ceiling investigation: workers x concurrency grid + httptools vs h11
	$(PYTHON) tools/ceiling_probe.py

.PHONY: mutation-check
mutation-check: ## Reintroduce each fixed bug and prove the regression tests fail on it
	$(PYTHON) tools/mutation_check.py

.PHONY: crosscheck
crosscheck: ## Cross-check the load generator against ApacheBench
	$(PYTHON) tools/crosscheck.py

.PHONY: smoke-wire
smoke-wire: ## Live validation of wire clients (binary tensor, gRPC, codecs, parameters.extra)
	$(PYTHON) tools/smoke_wire.py

.PHONY: smoke-test
smoke-test: ## One gated trial (wire-format stock-vs-binary, 1 round, C1)
	$(RUN) --domain wire-format --plan stock-vs-binary --rounds 1 --concurrency 1

# ------------------------------------------------------------- benchmark ---

.PHONY: run-wire-format
run-wire-format: ## All wire-format plans (~2 h)
	$(RUN) --domain wire-format --plan stock-vs-binary
	$(RUN) --domain wire-format --plan stock-vs-grpc
	$(RUN) --domain wire-format --plan binary-fallback
	$(RUN) --domain wire-format --plan mixed-negotiation

.PHONY: run-processing
run-processing: ## All processing plans (~4 h)
	$(RUN) --domain processing --plan frontend-vs-worker-gpu
	$(RUN) --domain processing --plan frontend-vs-worker-cpu
	$(RUN) --domain processing --plan worker-scaling
	$(RUN) --domain processing --plan concurrency-saturation
	$(RUN) --domain processing --plan batched-fallback

.PHONY: run-threading
run-threading: ## All threading plans (~4 h)
	$(RUN) --domain threading --plan gpu-thread-sweep
	$(RUN) --domain threading --plan gpu-spinning
	$(RUN) --domain threading --plan gpu-cudnn
	$(RUN) --domain threading --plan cpu-thread-factorial
	$(RUN) --domain threading --plan cpu-grpc-thread-scaling
	$(RUN) --domain threading --plan cpu-spinning
	$(RUN) --domain threading --plan cpu-affinity
	$(RUN) --domain threading --plan cpu-spin-tuning
	$(RUN) --domain threading --plan cpu-latency-c1

.PHONY: run-stability
run-stability: ## Soak tests (~1 h each)
	$(RUN) --domain stability --plan soak-60min

.PHONY: run-transport
run-transport: ## Transport plans (~30 min)
	$(RUN) --domain transport --plan queue-vs-zmq
	$(RUN) --domain transport --plan route-vs-direct

.PHONY: run-plan
run-plan: ## Run one plan: make run-plan DOMAIN=<domain> PLAN=<name>
	@test -n "$(DOMAIN)" && test -n "$(PLAN)" || \
		{ echo "usage: make run-plan DOMAIN=<domain> PLAN=<name>"; exit 2; }
	$(RUN) --domain $(DOMAIN) --plan $(PLAN)

.PHONY: run-all
run-all: run-wire-format run-processing run-threading run-stability run-transport ## Everything (~12 h)

.PHONY: resume
resume: ## Re-run every plan the checkpoint does not mark complete
	$(PYTHON) tools/resume.py

# ------------------------------------------------------------ benchmark ---
# Docker-mode engine (bench/): wizard, all-framework suites, parity gates.

BENCH = cd $(REPO_DIR) && $(PYTHON) -m bench.cli

.PHONY: benchmark
benchmark: ## Interactive benchmark wizard (alias: make run)
	$(BENCH) benchmark

.PHONY: run
run: benchmark ## Alias for the interactive wizard

.PHONY: benchmark-cpu
benchmark-cpu: ## Full suite, all 4 CPU frameworks, CPU image
	$(BENCH) run --engine $(ENGINE) --target cpu --framework all

.PHONY: benchmark-gpu
benchmark-gpu: ## Full suite, GPU onnx, GPU image (CUDA provider guard on)
	$(BENCH) run --engine $(ENGINE) --target gpu --framework onnx

.PHONY: benchmark-one
benchmark-one: ## One framework: make benchmark-one TARGET=cpu FRAMEWORK=sklearn
	@test -n "$(TARGET)" && test -n "$(FRAMEWORK)" || \
		{ echo "usage: make benchmark-one TARGET=<cpu|gpu> FRAMEWORK=<name>"; exit 2; }
	$(BENCH) run --target $(TARGET) --framework $(FRAMEWORK)

.PHONY: dry-run
dry-run: ## Validate images + zoo checksums + boot server + parity gate (no load)
	$(BENCH) dry-run --target $(if $(TARGET),$(TARGET),cpu) \
		--framework $(if $(FRAMEWORK),$(FRAMEWORK),all) \
		$(if $(IMAGE),--image $(IMAGE),)

.PHONY: compare
compare: ## Statistical regression: make compare BASELINE=<dir> CANDIDATE=<dir>
	@test -n "$(BASELINE)" && test -n "$(CANDIDATE)" || \
		{ echo "usage: make compare BASELINE=<dir> CANDIDATE=<dir>"; exit 2; }
	$(BENCH) compare $(BASELINE) $(CANDIDATE)

.PHONY: models
models: ## Materialize the model zoo on demand (reuse-if-present)
	cd $(REPO_DIR) && $(PYTHON) -c "from bench import zoo; \
		from pathlib import Path; \
		zoo.materialize(Path('.'), ['sklearn','xgboost','lightgbm','onnx'], \
		['small','medium','large'])"

.PHONY: models-clean
models-clean: ## Remove the auto-materialized model zoo (regenerates on demand)
	cd $(REPO_DIR) && $(PYTHON) -c "from bench import zoo; \
		from pathlib import Path; zoo.clean(Path('.'))"

# ---------------------------------------------------------- diagnose ---

.PHONY: diagnose
diagnose: ## Run 6-layer diagnostic (system, process, CPU, memory, latency, GPU)
	$(BENCH) diagnose --engine $(ENGINE) $(if $(CONTAINER),--container $(CONTAINER),)

.PHONY: diagnose-quick
diagnose-quick: ## Quick diagnostic (skip py-spy/memray profiling)
	$(BENCH) diagnose --skip-profile

.PHONY: reference
reference: ## 3-point reference: raw ORT floor, MLServer, Triton ceiling
	$(BENCH) reference --engine $(ENGINE) $(if $(TIER),--tier $(TIER),) $(if $(GPU),--gpu,)

.PHONY: reference-gpu
reference-gpu: ## Reference with GPU (floor + Triton on CUDA)
	$(BENCH) reference --gpu

# ------------------------------------------------------ analysis & gates ---

.PHONY: summarize
summarize: ## Re-run session summarization over all collected results
	@for s in results/*/*/; do \
		[ -f "$$s/SUMMARY.json" ] || continue; \
		$(PYTHON) tools/summarize.py --session "$$s" \
			--sampler "$$s/samples.jsonl" || true; \
	done

.PHONY: analyze
analyze: ## Paired statistical analysis (bootstrap CI) on collected sessions
	$(PYTHON) tools/analyze.py results/

.PHONY: validate
validate: ## Compliance-check every collected session (gates, receipts, spread)
	$(PYTHON) tools/validate.py results/

.PHONY: report
report: ## Generate reports/benchmark-report.md from collected sessions
	$(PYTHON) tools/report.py results/

.PHONY: checkpoint
checkpoint: ## Show which plans are complete / pending
	$(PYTHON) tools/checkpoint.py

.PHONY: status
status: checkpoint ## Alias for checkpoint

# ----------------------------------------------------------- maintenance ---

.PHONY: generate
generate: ## Regenerate all domain variants.json and plans from generate.py
	@for d in benchmarks/*/; do \
		[ -f "$$d/generate.py" ] && $(PYTHON) "$$d/generate.py"; \
	done

.PHONY: cleanup
cleanup: ## Report stale sweeps / pod storage / local samplers
	$(PYTHON) tools/cleanup.py --all

.PHONY: lint
lint: ## flake8 over bench, tools and generators
	$(PYTHON) -m flake8 bench tools benchmarks/*/generate.py

.PHONY: test
test: ## Unit and lightweight integration tests (no cluster, no docker load)
	$(PYTHON) -m pytest tools/tests bench/tests -q

.PHONY: clean
clean: ## Stop/remove benchmark containers, temp dirs, stale lockfiles, zoo
	-$(PYTHON) -c "from bench.docker_server import clean_stale_containers; \
		print('removed:', clean_stale_containers())"
	rm -rf models-zoo
	rm -f reports/benchmark-report.md
	find results -name "*.partial" -delete 2>/dev/null || true
	find . -name "*.json.tmp" -delete 2>/dev/null || true
