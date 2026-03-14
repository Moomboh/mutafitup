# mutafitup

**Mu**lti **Ta**sk **Fi**ne **Tu**ning of **P**rotein language models

`mutafitup` is a reproducible and highly customizable pipeline for evaluating multi-task parameter-efficient fine-tuning (PEFT) of protein language models. Orchestrated with **Snakemake**, it currently spans 16 diverse downstream prediction tasks (e.g., secondary structure, localization, thermal stability, ligand binding) and uses **ESM-C 300M** as the primary backbone. 

The pipeline is designed with a focus on full reproducibility and is easily extensible for other datasets, new task combinations, or different PLM backbones. As a built-in feature, it also supports cross-task sequence leakage mitigation via clustering and resplitting.

## 🛠 Setup & Installation

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

## 🚀 Running the Pipeline

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

## ⚙️ Configuration & Environment Pinning

- **Configuration:** Stored in `config/config.yml` (production) and `config/config_dev.yml` (development). 
- **Profiles:** Found under `workflow/profiles/`, managing the execution environments (local vs. slurm/cluster).
- **Environment Pinning:** If you modify `environment.yml` or any Snakemake env files in `workflow/envs/`, you can regenerate the cross-platform `.pin.txt` lockfiles by running:
  ```bash
  just pin-envs
  ```

## 💻 Development Commands

We use `ruff` for formatting/linting, `pyright` for static type checking, and `pytest` for unit testing. 

```bash
just format        # Format code with ruff
just lint          # Run pyright
just test          # Run test suite and check coverage
just clean         # Remove cache directories and temporary files
```