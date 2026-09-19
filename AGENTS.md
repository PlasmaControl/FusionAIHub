# Repository Guidelines

## Project Structure & Module Organization

FusionAIHub (FAITH) develops multimodal models and analysis tools for tokamak plasma data.

- `src/tokamak_foundation_model/`: data processing, modality models, training, IGNITE codecs/dynamics, and end-to-end models.
- `src/labelmaker/`: feature resolution, event labeling, calibration, and validation.
- `src/ideate/`: shot analysis, retrieval, experiment design, and CLI/MCP interfaces.
- `src/faith/`: package version metadata; the distribution includes all four packages.
- `tests/`: component suites, including `e2e/`, `ignite/`, `labelmaker/`, and `ideate/`.
- `scripts/`: training, evaluation, data preparation, and platform-specific SLURM launchers. `configs/` holds shared configuration; `docs/` holds architecture notes; `analysis/` contains figure-generation scripts.

## Build, Test, and Development Commands

Run commands from the repository root. Pixi manages Python 3.11 and installs the project in editable mode.

- `pixi install`: install the default CUDA environment.
- `pixi install -e frontier`: install the Frontier environment; follow `README.md` for FlashAttention setup.
- `pixi run pytest tests/e2e/test_lora.py -q`: run a focused model test.
- `pixi run -e labelmaker pytest tests/labelmaker -q`: run labeling tests with their environment dependencies.
- `pixi run -e ideate ideate-test`: run the IDEATE suite.
- `pixi run -e ideate ideate --help`: inspect the analysis CLI.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Follow surrounding type annotations and docstrings. Ruff is configured in `pyproject.toml` with an 88-character line limit. Keep model logic in `src/` and execution wrappers in `scripts/`.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Add focused regression tests for behavior changes; use deterministic seeds for numerical tests. No coverage percentage is configured. The `real_data` marker identifies tests requiring external stores or weights; use `-m "not real_data"` to exclude marked tests. Check individual tests for additional GPU and opt-in environment requirements.

## Commit & Pull Request Guidelines

Recent commits use component prefixes, such as `labelmaker: ...` and `ideate: ...`. Keep subjects concrete and changes focused. In PRs, explain the behavior change, link relevant issues, report test commands/results, and identify data or hardware requirements. Include plots when reconstruction or prediction quality changes.

## Data & Configuration

Treat production data stores as read-only. Keep credentials, datasets, checkpoints, and generated run artifacts out of commits. Use configuration or environment variables for machine-specific paths, and review dependency changes together with `pixi.lock`.
