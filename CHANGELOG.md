# Changelog

## [0.2.0] - 2026-04-05

### Fixed
- **A1** — Credential routing: `HF_TOKEN` fallback added to `inference.py` so HuggingFace deployments work without `OPENAI_API_KEY`
- **A2** — Grader score clamped to `[0.0, 1.0]` — was returning up to 1.05 due to missing `min(1.0, ...)` in `_compute_grader_score`
- **A3** — RNG seeded deterministically per task: `random.seed(task_id * 1000)` ensures same `reset(seed=N)` produces identical persona every run
- **A4** — `openenv.yaml` expanded with version, description, max_steps, health_check, env_variables, resources
- **C1** — `SCHEME_PRIORITY` order corrected to `PMAY > MGNREGS > PMKVY` — was `MGNREGS` first, contradicting system prompt benefit hierarchy
- **C2** — `get_optimal_scheme()` docstring updated to match corrected priority order
- **C3** — Age bounds truthiness bug fixed in `get_eligible_schemes()`: `if rules.get("age_min")` → `if rules.get("age_min") is not None`
- **C4** — Warning comments added above each extended scheme (PM_SYM, AYUSHMAN_BHARAT, E_SHRAM, NFSA, PMMVY) explaining they are not reachable from tasks 1–5
- **D1** — `asyncio` and `copy` imports added; `_state_lock = asyncio.Lock()` added as class variable
- **D2** — Task 5 age conflict randomised: `aadhaar_age = self_reported_age + random.randint(1, 3)` — was hardcoded as 35/36 every episode
- **D3** — `asyncio.Lock` guards `_shared_state` access to prevent concurrent request corruption
- **D4** — `copy.deepcopy(self._obs)` in `step()` prevents aliased mutation of shared observation state
- **D5** — `_finalize_step` timeout check fixed: `>=` → `>` to prevent overwriting correct terminal actions on step 19
- **D6** — Task 3 rejection now requires `income` in `known_profile` before scoring — closes always-reject exploit
- **D7** — Blind escalate base score reduced: `0.75` → `0.25` — agent must verify PAN card to score well on Task 4
- **D8** — Fallback grader score fixed: `reward >= 5.0` now maps to `0.75` (not `1.0`) — correct rejection and correct escalation now distinguishable
- **D9** — JSON extraction uses last `{...}` match (`re.findall` + `[-1]`) to avoid extracting from inside `<think>` blocks
- **E1** — `/health` endpoint added to `app.py` — Docker HEALTHCHECK was permanently failing
- **E4** — `requirements.txt` version-pinned with upper bounds to prevent Pydantic v1/v2 incompatibility
- **E5** — `_wait_for_server()` added to `benchmark_runner.py` — polls `/health` before starting inference
- **E6** — Structured `SCORE_JSON` lines emitted by `inference.py`; `benchmark_runner.py` parses these as primary source instead of fragile regex
- **E7** — `pyproject.toml` dependencies pinned to match `requirements.txt`
- **F2** — `INFERENCE_TEMPERATURE` env var added — temperature is now configurable (default `0.0` for eval, set `> 0` for training)
- **H2/H5** — `client.py` cleaned: removed `echoed_message` / `message_length` template fields, now correctly maps Scheme Env observation fields
- **I4** — `MAX_TOKENS` env var added — token limit now configurable via environment

### Added
- `tests/test_environment.py` — unit tests covering eligibility grading, determinism, and grader score range (pytest)
- `.dockerignore` — excludes `__pycache__`, reports, venv, and build artefacts from Docker build context
- `CHANGELOG.md` — this file

## [0.1.0] - 2026-04-04

### Added
- Initial release: 5-task Indian CSC welfare officer RL environment
- Tasks: Scheme Discovery, Missing Data, Boundary Fraud Detection, Escalation Dilemma, Document Conflict
- Dense per-step rewards with efficiency-adjusted grader scores
- Baseline inference script with `<think>` + JSON format
- Benchmark runner with CSV leaderboard and matplotlib report generation