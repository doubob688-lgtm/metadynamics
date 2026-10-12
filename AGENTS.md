# AGENTS.md

## Project scope

This workspace is for metadynamics research into the reactivity of a novel diazirine system. Keep implementations scientifically reproducible, modular, and easy to adapt as the reaction model and collective variables evolve.

## Python environment

- Use Python 3.12 from the Conda environment at:
  `/home/bob/miniconda3/envs/metad312/bin/python`
- Invoke that interpreter directly by default for scripts, tests, and package checks.
- Do not install packages into the system Python. Add or change dependencies only when needed, and record them in the project's environment or dependency files when those files exist.

## Repository structure

- `src/` contains reusable project source code, including simulation components, calculator adapters, collective-variable definitions, and analysis utilities.
- `model/` contains local model files. Use the MACE OMol model stored in this directory as the default potential.
- `result/` contains generated calculation and analysis results. Organize outputs into clearly named run-specific subdirectories and avoid mixing generated data with source code.
- `script/` contains runnable Python (`.py`) and Bash (`.sh`) scripts used to launch simulations, analyses, setup tasks, and other workflows.
- Keep substantial reusable logic in `src/`; scripts in `script/` should remain thin entry points that import and configure that logic.
- Write generated outputs to `result/` by default unless the user specifies another destination.

## Metadynamics architecture

- Implement metadynamics through the ASE–PLUMED integration.
- Use the MACE OMol model from `model/` as the default calculator unless the user or a workflow explicitly selects another MACE model or PySCF.
- Keep simulation setup, collective-variable definitions, PLUMED input, calculator construction, and trajectory analysis separated where practical.
- Design metadynamics workflows so the ASE calculator backend can be switched between MACE and PySCF without rewriting the sampling logic.
- Avoid backend-specific assumptions in shared metadynamics code. Put MACE- and PySCF-specific configuration behind small, explicit calculator factories or adapters.
- Preserve units carefully across ASE, PLUMED, MACE, and PySCF, and document any conversion that is not self-evident.
- Set random seeds where supported and retain the inputs, configuration, and metadata needed to reproduce a run.

## Compute defaults

- Prefer CUDA/GPU execution whenever relevant and supported.
- For PySCF workflows, prefer GPU4PySCF when a compatible GPU and installation are available.
- For MACE workflows, select a CUDA device when PyTorch reports CUDA availability.
- Detect capability at runtime rather than assuming that a GPU is present.
- Automatically fall back to the corresponding CPU implementation when CUDA, GPU4PySCF, or another GPU dependency is unavailable. The fallback should be clear in logs and must not otherwise change the intended scientific method.
- Do not fail solely because GPU acceleration is unavailable unless the user explicitly requests GPU-only execution.

## Validation

- Before launching an expensive calculation, run a short smoke test that validates imports, calculator construction, ASE–PLUMED coupling, device selection, and basic force/energy evaluation.
- Keep CPU smoke tests small enough to run locally after an automatic fallback.
- Check that generated PLUMED inputs and collective variables match the intended atom indexing and periodic-boundary conventions.
- Never fabricate successful scientific results. Report numerical instability, unsupported GPU paths, convergence failures, and fallback behavior explicitly.
