# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Important:** Before starting work, read [AGENTS.md](AGENTS.md) for the project's development guidelines, style rules, NPU-specific constraints, and review checklist.

## Repository Overview

vLLM Ascend (`vllm-ascend`) is a hardware plugin that runs vLLM on Huawei Ascend NPUs. It integrates with upstream vLLM through the pluggable hardware interface defined in [vLLM Hardware Plugin RFC](https://github.com/vllm-project/vllm/issues/11162) rather than by forking vLLM or adding model files directly.

Key branch mapping:

- `main` tracks vLLM `main`.
- `releases/vX.Y.Z` is the development branch for the matching vLLM release tag (e.g. `releases/v0.23.0`).

This repository is on `releases/v0.23.0`.

## Common Development Commands

### Setup

```bash
# Install the package and all dev/test dependencies.
pip install -e .[dev]

# Or install requirements directly:
pip install -r requirements-dev.txt
```

`setup.py` compiles the C++ extension `vllm_ascend.vllm_ascend_C` via CMake. The build auto-detects the SoC from `npu-smi` unless `SOC_VERSION` is set. Set `COMPILE_CUSTOM_KERNELS=0` only for CPU-only UT environments.

### Lint and Format

```bash
# Run the same lint/format check used in CI.
bash format.sh ci

# Run interactively (local pre-commit stage).
bash format.sh

# Run pre-commit directly.
pre-commit run --all-files --hook-stage manual
```

Pre-commit covers ruff, codespell, typos, clang-format, markdownlint, actionlint, shellcheck, and several repo-specific checks in `tools/`.

### Type Checking

```bash
# Run mypy for the local Python version.
tools/mypy.sh

# Run mypy as CI does for Python 3.10, 3.11, and 3.12.
tools/mypy.sh 1 3.10
tools/mypy.sh 1 3.11
tools/mypy.sh 1 3.12
```

### Running Tests

```bash
# Run a single unit test file.
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run a single test function.
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run all CPU-routable unit tests (no NPU required; conftest.py mocks torch_npu).
pytest -sv tests/ut

# Run NPU-specific E2E tests (requires Ascend hardware).
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py::test_default_full_and_piecewise_res_consistency
```

Test routing rules:

- `tests/ut/<module>/` → CPU runner by default.
- `tests/ut/<module>/a2/` → A2 NPU x1; `a2_2/` → A2 NPU x2.
- `tests/ut/<module>/a3_2/` → A3 NPU x2; `a3_4/` → A3 NPU x4.
- `tests/ut/<module>/310p/` → 310P NPU x1.
- E2E paths under `tests/e2e/pull_request/{one_card,two_card,four_card}/` are matched by card count.

The selective test scope is configured in `.github/workflows/scripts/test_config.yaml`.

### Build Commands

```bash
# Rebuild the C++ extension after source changes.
pip install -e . --force-reinstall --no-deps

# Or build the CANN custom operators directly (advanced).
bash csrc/build.sh --pkg --soc=ascend910b -j16 -O3
```

### Documentation

```bash
cd docs
make html
```

## High-Level Architecture

### Plugin Entry Points

`vllm_ascend/__init__.py` exports the plugin hooks that vLLM discovers:

- `register()` → returns `"vllm_ascend.platform.NPUPlatform"`.
- `register_connector()`, `register_model_loader()`, `register_service_profiling()`, `register_model()` → register distributed/model loader/profiling/model extensions.
- `_ensure_global_patch()` applies process-wide platform patches in engine-core subprocesses.

### NPU Platform

`vllm_ascend/platform.py` defines `NPUPlatform`, the vLLM `Platform` implementation for Ascend. Key responsibilities:

- Declares device type `npu`, dispatch key `PrivateUse1`, and supported quantization methods (`ascend`, `compressed-tensors`, `fp8`, `deepseek_v4_fp8`).
- `pre_register_and_update()` and `check_and_update_config()` adjust vLLM config for NPU (CUDA graph sizes, compilation config, sleep mode, etc.).
- Provides the custom Inductor pass manager and compiler backend (`vllm_ascend.compilation`).

### Patch System

vLLM Ascend modifies upstream behavior through monkey-patches in `vllm_ascend/patch/`:

- `patch/platform/` — applied early via `vllm_ascend.utils.adapt_patch(is_global_patch=True)` in `NPUPlatform.pre_register_and_update()` and plugin entry points. These patches affect scheduler, config validation, distributed setup, and model-runner selection before workers start.
- `patch/worker/` — applied per worker via `adapt_patch()` in each worker's `__init__`. These patches modify model-specific forward passes, custom op behavior, and weight handling.

Patch `__init__.py` documents every patch, its upstream target, and the planned upstream path. New patches require strict architectural review per AGENTS.md.

### Model Runners

There are three model runners for different vLLM/Ascend variants:

- `vllm_ascend/worker/model_runner_v1.py` — vLLM v1 model runner, the default. Extends `GPUModelRunner` with NPU-specific attention, KV cache, and ACL graph handling.
- `vllm_ascend/worker/v2/model_runner.py` — vLLM v2 model runner. Enabled only when `VLLM_USE_V2_MODEL_RUNNER=1`. It also subclasses `GPUModelRunner` but has a narrower feature set (no context parallelism or dynamic EPLB).
- `vllm_ascend/_310p/model_runner_310p.py` — dedicated runner for Ascend 310P devices.

`vllm_ascend/patch/platform/patch_use_v2_model_runner.py` removes upstream v2 whitelisting and delegates the decision to the `VLLM_USE_V2_MODEL_RUNNER` environment variable.

### Attention Backends

`vllm_ascend/attention/` implements vLLM v1 attention backends:

- `attention_v1.py` registers `AscendAttentionBackend` for `AttentionBackendEnum.CUSTOM` (`ASCEND`).
- Variants live in `mla_v1.py`, `sfa_v1.py`, `dsa_v1.py`, `fa3_v1.py`, and `context_parallel/`.
- `AscendAttentionBackend.get_impl_cls()` returns the context-parallel implementation when context parallelism is enabled, otherwise the standard implementation.

### Custom Operators

`vllm_ascend/ops/` provides NPU replacements for upstream vLLM layers (RMSNorm, linear layers, rotary embeddings, MLA, MoE, etc.). `vllm_ascend.utils.register_ascend_customop()` registers them in `CustomOp` so upstream model code automatically uses the Ascend implementations.

The C++ custom-operator extension `vllm_ascend.vllm_ascend_C` is built from `csrc/` and bound in `csrc/torch_binding.cpp`.

### Compilation and ACL Graphs

`vllm_ascend/compilation/` provides:

- `compiler_interface.py` — `AscendCompiler` Inductor backend and graph-fusion pass plumbing.
- `graph_fusion_pass_manager.py` — pass manager for pattern-based fusions.
- `passes/` — individual FX/Inductor fusion passes.
- `acl_graph.py` — ACL graph capture/replay integration used for decode-batch acceleration.

### Ascend Configuration

`vllm_ascend/ascend_config.py` defines `AscendConfig`, populated from vLLM's `--additional-config`. It groups tuning knobs such as compilation config, fusion config, fine-grained TP config, EPLB config, weight prefetch config, and profiling chunk config. New `additional_config` options should be added here, not as standalone CLI flags.

### Environment Variables

All environment variables are centralized in `vllm_ascend/envs.py` inside the `env_variables` dict. When adding a new variable, document it there, follow the `VLLM_ASCEND_*` naming convention, and ensure it is reviewed per AGENTS.md.

### Worker and Distributed

`vllm_ascend/worker/worker.py` (`NPUWorker`) extends `WorkerBase`. It initializes the platform patch, registers custom ops, sets up CPU binding, initializes the Ascend model-parallel environment, and owns the model runner.

`vllm_ascend/distributed/` contains HCCL/HCCL-plugin communicators, EP/load-balancer helpers, and KV/weight transfer registration.

### 310P Path

`vllm_ascend/_310p/` contains a separate model runner, worker, attention, and quantization path for Ascend 310P devices. Changes here should be verified on 310P hardware.

## Notes for Code Changes

- Follow the patch-vs-inheritance guidance in AGENTS.md: prefer upstream contribution or inheritance; use patches only when necessary.
- Avoid `tensor.item()` on NPU tensors in hot paths because it forces CPU-NPU synchronization.
- Keep new environment variables in `vllm_ascend/envs.py`.
- Add unit tests in `tests/ut/` and integration/E2E tests in `tests/e2e/` for new functionality.
- Commit messages must follow Conventional Commits and include a sign-off (`git commit -s`).
- PR titles must use one of the prefixes defined in CI: `[BugFix]`, `[Performance]`, `[Test]`, `[CI]`, `[Feature]`, `[Doc]`, `[Misc]`, `[Community]`, or `[Refactor]`.
- Run `bash format.sh ci` before pushing.
