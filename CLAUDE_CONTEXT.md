# SYSTEM ARCHITECTURE & AI DEVELOPER DIRECTIVES
**Project:** Government Scheme Execution Environment
**Target:** Meta OpenEnv Hackathon x Scaler School of Technology
**Version:** 2.0 (Streamlined MVP)

## 1. PROJECT OVERVIEW
We are building an open-source Reinforcement Learning (RL) environment that simulates the workflow of an Indian Government Welfare Officer. The environment trains an LLM-based agent to interview applicants, gather missing data, and correctly enroll them in welfare schemes (e.g., PMKVY, MGNREGS) or safely reject them.

This environment will be evaluated by automated bots using an `inference.py` script. The environment must be strictly compliant with the OpenEnv specification, containerized via Docker, and feature exactly three tasks of increasing difficulty (Easy, Medium, Hard) graded between `0.0` and `1.0`.

---

## 2. TECHNOLOGY STACK
* **Core Framework:** `openenv-core` (Python 3.11)
* **Data Validation:** `pydantic`
* **API/Server:** `fastapi`, `uvicorn`
* **Deployment:** Docker (Single-stage `python:3.11-slim`, exposing **Port 7860** strictly)
* **Inference Agent:** `openai` Python client routing to Hugging Face models via standard HTTP APIs.

---

## 3. STRICT ARCHITECTURAL CONSTRAINTS
**WARNING TO AI ASSISTANT:** We previously attempted a complex 12-file, 6-stage, 20-action architecture. It resulted in catastrophic routing failures and Docker crashes. **DO NOT over-engineer.** Adhere strictly to the following constraints:

### 3.1. Action Space (Fixed at 5 Actions)
The agent may only execute the following actions via `action_type`:
1. `ask_question` (Requires `value`: e.g., "age", "income", "occupation")
2. `request_document` (Requires `value`: e.g., "Aadhaar", "Bank Passbook")
3. `approve_scheme` (Requires `value`: e.g., "PMKVY", "MGNREGS")
4. `reject_applicant` (Requires `value`: reason for rejection)
5. `escalate` (Requires no value; used when the agent is stuck)

### 3.2. State & Termination (The 12-Step Rule)
* The environment must track `step_count`. 
* **MAX_STEPS is strictly 12.** * If `step_count >= 12`, the environment must force `done = True`, set `is_terminated = True`, and apply a negative reward penalty (Timeout).

### 3.3. Pydantic Data Models (`models.py`)
Do not alter this schema without explicit human permission:
* `Action`: `action_type` (str), `value` (Optional[str])
* `Observation`: `known_profile` (Dict), `missing_data` (List), `notification` (Optional[str]), `is_terminated` (bool)

### 3.4. File Structure
* **Logic:** `server/scheme_env_environment.py` (Contains the `Environment` class, `step`, and `reset` logic).
* **Models:** `models.py` (Root level).
* **Entrypoint:** `server/app.py` (Contains `create_app` and uvicorn configuration on Port 7860).
* **Evaluation:** `inference.py` (Root level).

---

## 4. THE 3-TASK PROGRESSION (EVALUATION CRITERIA)
To pass the Hackathon's Automated Validation Gate, the environment must programmatically generate 3 distinct scenarios and grade the agent's performance at the end of the episode. 

### TASK 1: Scheme Discovery (Difficulty: Easy)
* **Initial State:** Applicant has complete data (e.g., Age 26, Mason, Income 4000).
* **Objective:** Agent must recognize the profile, optionally ask 1 confirmation question, and execute `approve_scheme` for "PMKVY".
* **Grader Logic:** * `1.0` -> Agent successfully approves the optimal scheme.
  * `0.5` -> Agent approves a technically eligible but sub-optimal scheme.
  * `0.0` -> Agent rejects the applicant, approves an ineligible scheme, or times out.

### TASK 2: Missing Data Identification (Difficulty: Medium)
* **Initial State:** Applicant is eligible for MGNREGS, but critical variables (e.g., "has_aadhaar", "occupation") are missing from `known_profile` and listed in `missing_data`.
* **Objective:** Agent must use `ask_question` or `request_document` to move items from `missing_data` to `known_profile` *before* taking a final action.
* **Grader Logic:**
  * `1.0` -> Agent clears the `missing_data` array and approves the correct scheme.
  * `0.1 to 0.9` -> Partial credit based on the percentage of missing data successfully collected before approval.
  * `0.0` -> Agent attempts to approve a scheme *before* clearing critical `missing_data` (Premature Approval violation).

### TASK 3: Conflict Resolution & Rejection (Difficulty: Hard)
* **Initial State:** Applicant profile contains disqualifying criteria (e.g., Age 14, or Income 500,000).
* **Objective:** Agent must evaluate the profile, recognize that no scheme applies, and execute `reject_applicant` with a valid reason.
* **Grader Logic:**
  * `1.0` -> Agent executes `reject_applicant`.
  * `0.0` -> Agent executes `approve_scheme` for ANY scheme (Severe Safety Violation).

---

## 5. REWARD SHAPING
The `step()` function must return dense, step-by-step mathematical rewards to train the RL agent effectively. Follow this exact matrix:
* **+1.0:** Asking a valid question from `missing_data`.
* **-1.0:** Asking an invalid, redundant, or already-answered question.
* **+10.0:** Correctly approving an eligible optimal scheme (Terminal).
* **+5.0:** Correctly rejecting an ineligible applicant (Terminal).
* **-5.0:** Approving an ineligible scheme (Terminal).
* **-2.0:** Hitting the 12-step timeout limit (Terminal).

---

## 6. INFERENCE SCRIPT SPECIFICATION (`inference.py`)
The `inference.py` script must exist in the root folder and adhere to these hackathon rules:
1. Must read `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables using `os.getenv()`.
2. Must instantiate the standard `openai.OpenAI` client.
3. Must execute a loop connecting to `http://localhost:7860`.
4. Must complete within 20 minutes.
5. Must explicitly print the Grader scores (0.0 to 1.0) for Task 1, Task 2, and Task 3 at the end of execution.

---

## 7. AI DEVELOPER DIRECTIVES (Rules of Engagement)
As the AI assisting in building this project, you must:
1. **Never use placeholder data in final files.** All personas must be realistic Indian welfare profiles.
2. **Never alter `Dockerfile` or `openenv.yaml`** unless explicitly commanded to do so. The container configuration is currently passing validation.
3. **Write Defensive Code:** In `step()`, ensure that if the LLM hallucinates an `action_type` that does not exist, the environment catches it, applies a `-1.0` reward, returns a helpful `notification` to guide the LLM, and does *not* crash the server.
4. **Test Before Advancing:** When building the Tasks, provide the code for Task 1 and ensure the user tests it before generating the code for Task 2.