# mutafitup

**Mu**lti **Ta**sk **Fi**ne **Tu**ning of **P**rotein language models

`mutafitup` is a reproducible and highly customizable pipeline for evaluating multi-task parameter-efficient fine-tuning (PEFT) of protein language models (pLMs). It was developed as a bachelor's thesis project at TUM and LMU Munich, carried out at [RostLab](https://rostlab.org/).

Orchestrated with **[Snakemake](https://snakemake.github.io/)**, the pipeline covers 16 diverse downstream prediction tasks spanning secondary structure, subcellular localization, thermal stability, intrinsic disorder, relative solvent accessibility, and ligand binding-site prediction. It includes built-in cross-task sequence leakage mitigation via clustering and resplitting, and supports four training strategies: heads-only, single-task LoRA, AccGrad-LoRA, and Align-LoRA.

[**ESM-C 300M**](https://www.evolutionaryscale.ai/papers/esm-c-a-bridge-between-sequence-and-structure-for-protein-language-models) was used as the primary backbone in the thesis. The pipeline architecture already supports multiple pLM backbones, including [**ESM-2**](https://www.science.org/doi/10.1126/science.ade2574), **ESM-C**, and [**ProtT5**](https://doi.org/10.1109/TPAMI.2021.3095381), and is easily extensible to others.

## Pretrained Models & Datasets

The fine-tuned models and benchmark datasets produced by this pipeline are available on Hugging Face:

- **[Moomboh/ESMC-300M-mutafitup](https://huggingface.co/Moomboh/ESMC-300M-mutafitup)** -- Multi-task LoRA fine-tuned ONNX models and PyTorch checkpoints derived from ESM-C 300M. Includes 45 training runs across 4 training strategies (heads-only, single-task LoRA, AccGrad-LoRA, Align-LoRA).
- **[Moomboh/mutafitup-datasets](https://huggingface.co/datasets/Moomboh/mutafitup-datasets)** -- 16 leakage-controlled protein prediction benchmarks in Parquet format. All datasets have been resplit to reduce cross-dataset sequence similarity-based data leakage while preserving upstream benchmark test splita (except for meltome).

## Project Overview

The repository consists of three main components:

**Snakemake pipeline** (`workflow/`, `src/`, `config/`)
The core of the project. Handles data fetching and preparation, cross-dataset leakage-free resplitting, model training across all four strategies, prediction, evaluation, and plotting. The Python library in `src/mutafitup/` provides the model architecture (multi-task backbone + task heads), dataset classes, training logic, metrics, ONNX export, and evaluation utilities.

**Inference app** (`app/`)
A cross-platform protein property prediction UI built with [Dioxus](https://dioxuslabs.com/) in Rust. Compiles to both a web app (WASM + onnxruntime-web) and a native desktop app. See the [Inference App](#inference-app) section below.

**TypeScript inference packages** (`packages/`)
Three npm packages providing JavaScript/TypeScript APIs for ONNX inference with exported mutafitup models: `mutafitup-common` (shared preprocessing and tokenization), `mutafitup-node` (Node.js inference via onnxruntime-node), and `mutafitup-web` (browser inference via onnxruntime-web).

### Repository Structure

```
├── src/mutafitup/       # Core Python library (models, training, evaluation, export)
├── src/wfutils/         # Workflow utility helpers
├── workflow/
│   ├── rules/           # Snakemake rule definitions
│   ├── scripts/         # Per-rule entry-point scripts
│   ├── envs/            # Conda environment specs for individual rules
│   └── profiles/        # Execution profiles (local, local-dev, lrz-ai, lrz-ai-dev)
├── config/              # Pipeline configuration (production + development)
├── app/                 # Dioxus inference app (Rust, web + desktop)
├── packages/            # TypeScript inference packages (pnpm workspace)
├── tests/               # Pytest test suite
├── resources/           # Static resources (SOTA reference metrics)
└── scripts/             # Standalone utility scripts (env pinning, HF upload)
```

## Setup & Installation

The pipeline relies on Conda/Mamba for dependency management. For strict reproducibility, we provide explicitly pinned environments for `linux-64` and `osx-arm64` generated via `conda-lock`.

1. **Create the environment from the pinned lockfile:**
   *For macOS (Apple Silicon):*
   ```bash
   conda create --name mutafitup --file environment.osx-arm64.pin.txt
   ```
   *For Linux:*
   ```bash
   conda create --name mutafitup --file environment.linux-64.pin.txt
   ```
   *Alternatively, to resolve from scratch without pinning:*
   `mamba env create -f environment.yml`

2. **Activate the environment and install project-specific dependencies:**
   ```bash
   conda activate mutafitup
   just install-dev
   ```

## Running the Pipeline

Execution of the Snakemake workflow is streamlined via `just`. The `just` commands automatically route to the correct **Snakemake profiles** located in `workflow/profiles/`.

### Development Mode (Recommended for testing)
The `dev` mode uses the `local-dev` profile (pointing to `config/config_dev.yml`), which runs smaller models and heavily subsamples the datasets to ensure the pipeline executes quickly end-to-end.

```bash
just dry-run-dev   # Preview the execution plan
just run-dev       # Execute the pipeline locally
```

### Production Mode
Production mode uses the `local` profile (pointing to `config/config.yml`) to train the primary model (**ESM-C 300M**) across the full datasets.

```bash
just dry-run       # Preview the full production execution plan
just run           # Execute the full pipeline locally
```

### Cluster Execution (LRZ AI Systems)
For large-scale training, the repository includes SLURM-ready profiles tuned for the Leibniz Supercomputing Centre (LRZ) AI systems.
- **`lrz-ai`**: Executes the full production configuration on the cluster using SLURM job submission.
- **`lrz-ai-dev`**: Executes the subsampled development configuration on the cluster.

To run the pipeline on the cluster, bypass `just` and invoke Snakemake directly with the desired profile:
```bash
snakemake --profile workflow/profiles/lrz-ai
```

## Inference App

The `app/` directory contains a cross-platform inference application for running predictions with exported ONNX models. It is built with [Dioxus](https://dioxuslabs.com/) in Rust and supports both web (WASM) and native desktop targets.

**Features:**
- FASTA sequence input (paste or file upload)
- Model selection from exported model manifests
- Multiple ONNX execution providers (CPU, CUDA, TensorRT, CoreML, WebGPU, and more)
- Batch inference with progress tracking
- Per-residue prediction visualization
- Side-by-side comparison of predictions from different models

### Running the app

The app requires the [Dioxus CLI](https://dioxuslabs.com/learn/0.7/getting-started) (`dx`). All commands are run from the `app/` directory.

**Web** (requires [miniserve](https://github.com/svenstaro/miniserve) to serve model files locally):
```bash
just web
```

**Desktop** (with optional GPU backend features):
```bash
just desktop            # CPU
just desktop cuda       # NVIDIA CUDA
just desktop tensorrt   # NVIDIA TensorRT
```

## Configuration & Environment Pinning

- **Configuration:** Stored in `config/config.yml` (production) and `config/config_dev.yml` (development). 
- **Profiles:** Found under `workflow/profiles/`, managing the execution environments (local vs. slurm/cluster).
- **Environment Pinning:** If you modify `environment.yml` or any Snakemake env files in `workflow/envs/`, you can regenerate the cross-platform `.pin.txt` lockfiles by running:
  ```bash
  just pin-envs
  ```

## Development

We use `ruff` for formatting/linting, `pyright` for static type checking, and `pytest` for unit testing.

```bash
just format        # Format code with ruff
just lint          # Run pyright
just test          # Run test suite and check coverage
just clean         # Remove cache directories and temporary files
```

## License

The source code is licensed under the [MIT License](LICENSE.md).

Fine-tuned models derived from ESM-C 300M are subject to the [EvolutionaryScale Cambrian Open License](THIRD_PARTY_LICENSES/CAMBRIAN_OPEN_LICENSE.md). See [NOTICE](NOTICE) for full attribution details.
