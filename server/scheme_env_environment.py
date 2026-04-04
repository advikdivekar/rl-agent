import random
import math  # NEW: Used for sigmoid normalization of reward
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .schemes import SCHEMES
from models import Action, Observation


# =========================================================
# NEW: ADVANCED REWARD SYSTEM UTILITIES
# =========================================================

# Converts raw reward into a bounded value (0–1)
# WHY: Prevents reward explosion and stabilizes training across different agents
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Task-specific weights for reward components
# WHY: Different tasks prioritize different behaviors
# Example: Task 4 prioritizes SAFETY over raw correctness
WEIGHTS = {
    1: (0.5, 0.2, 0.2, 0.1),  # Task 1 → correctness focused
    2: (0.4, 0.2, 0.3, 0.1),  # Task 2 → reasoning more important
    3: (0.5, 0.2, 0.2, 0.1),  # Task 3 → strict correctness
    4: (0.3, 0.1, 0.2, 0.4),  # Task 4 → safety critical
}


# Multi-dimensional reward computation
# WHY: Replaces simple win/lose reward with realistic evaluation
def _compute_final_reward(task, obs, step_count, persona, base_outcome):

    # ---------------- TASK SUCCESS ----------------
    # Primary objective: correct decision-making
    if base_outcome == "optimal":
        task_reward = 6
    elif base_outcome == "suboptimal":
        task_reward = 4
    elif base_outcome == "safe":
        task_reward = 3
    else:
        task_reward = -6  # Strong penalty for wrong decisions

    # ---------------- EFFICIENCY ----------------
    # Rewards fewer steps (discourages unnecessary exploration)
    ideal_steps = len(persona.get("missing_keys", [])) + 1
    efficiency = ideal_steps / max(step_count, 1)
    efficiency_reward = 3 * efficiency

    # ---------------- REASONING QUALITY ----------------
    # Rewards meaningful queries and penalizes useless ones
    reasoning_reward = (
        obs.metadata.get("relevant_queries", 0) * 0.5
        - obs.metadata.get("noise_queries", 0) * 0.7
        - obs.metadata.get("redundant_queries", 0) * 0.5
        + obs.metadata.get("critical_discoveries", 0) * 2
    )

    # ---------------- SAFETY ----------------
    # Critical for fraud detection and real-world systems
    safety_reward = 0
    if task == 4:
        if obs.metadata.get("critical_discoveries", 0) > 0:
            safety_reward += 2  # Reward detecting contradiction
        if base_outcome == "safe":
            safety_reward += 1  # Reward correct escalation

    # ---------------- WEIGHTED COMBINATION ----------------
    # Combine all components based on task importance
    w_task, w_eff, w_reason, w_safe = WEIGHTS[task]

    total_reward = (
        w_task * task_reward
        + w_eff * efficiency_reward
        + w_reason * reasoning_reward
        + w_safe * safety_reward
    )

    # ---------------- NORMALIZATION ----------------
    # Convert reward into stable score between 0 and 1
    final_score = sigmoid(total_reward / 10)

    return total_reward, round(final_score, 3)


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

# Maximum steps allowed per episode before forced termination
MAX_STEPS = 20


# Irrelevant fields injected to distract the agent
# WHY: Tests whether the agent can ignore unnecessary information
NOISE_FIELDS = [
    "marital_status",
    "state_of_residence",
    "number_of_children",
    "bank_name",
]


# Random values for noise fields
NOISE_VALUES = {
    "marital_status": ["married", "unmarried", "widowed", "divorced"],
    "state_of_residence": ["Maharashtra", "Uttar Pradesh", "Bihar", "Rajasthan", "Gujarat"],
    "number_of_children": ["0", "1", "2", "3", "4"],
    "bank_name": ["SBI", "PNB", "Bank of Baroda", "Canara Bank", "UCO Bank"],
}


# Only these fields are relevant for decision-making
VALID_QUERY_FIELDS = {"age", "income", "occupation", "has_aadhaar"}


# Employers used in contradiction scenario (Task 4)
CONTRADICTION_EMPLOYERS = [
    "Indian Railways", "BSNL", "Coal India", "State Bank of India",
    "ONGC", "BHEL", "HAL", "GAIL India",
]


# Inject random noise fields into profile
def _inject_noise(profile: dict) -> dict:
    chosen = random.sample(NOISE_FIELDS, k=random.randint(1, 3))
    for field in chosen:
        profile[field] = random.choice(NOISE_VALUES[field])
    return profile


# =========================================================
# PERSONA GENERATION
# =========================================================

# Generates randomized applicant profiles per task
# WHY: Prevents overfitting and ensures robustness
def generate_dynamic_persona(task_id: int) -> dict:

    if task_id == 1:
        age = random.randint(18, 35)
        income = random.randint(1000, 9999)
        occ = random.choice(["mason", "carpenter"])

        eligible = ["PMKVY"]
        if income < 6000 and 21 <= age <= 55:
            eligible.append("PMAY")

        return {
            "age": str(age),
            "income": str(income),
            "occupation": occ,
            "has_aadhaar": "True",
            "optimal_scheme": "PMKVY",
            "eligible_schemes": eligible,
        }

    elif task_id == 2:
        return {
            "age": str(random.randint(18, 60)),
            "income": str(random.randint(1000, 5000)),
            "occupation": "farm_labourer",
            "has_aadhaar": "True",
            "optimal_scheme": "MGNREGS",
            "eligible_schemes": ["MGNREGS"],
            "missing_keys": ["occupation", "has_aadhaar"],
        }

    elif task_id == 3:
        return {
            "age": str(random.randint(22, 34)),
            "income": str(random.randint(10001, 12000)),
            "occupation": random.choice(["mason", "carpenter"]),
            "has_aadhaar": "True",
            "optimal_scheme": None,
            "eligible_schemes": [],
            "_near_miss": True,
        }

    elif task_id == 4:
        return {
            "age": str(random.randint(22, 45)),
            "income": str(random.randint(2000, 8000)),
            "occupation": "student",
            "has_aadhaar": "True",
            "optimal_scheme": None,
            "eligible_schemes": [],
            "_contradictory": True,
            "_pan_employer": random.choice(CONTRADICTION_EMPLOYERS),
        }

    else:
        raise ValueError(f"Unknown task_id: {task_id}")


# =========================================================
# OBSERVATION CREATION
# =========================================================

def _make_fresh_obs(task: int, persona: dict) -> Observation:
    profile = {
        "age": persona["age"],
        "income": persona["income"],
    }

    if task in [1, 3, 4]:
        profile["occupation"] = persona["occupation"]
        profile["has_aadhaar"] = persona["has_aadhaar"]

    _inject_noise(profile)

    return Observation(
        known_profile=profile,
        missing_data=list(persona.get("missing_keys", [])),
        notification="",
        is_terminated=False,
        reward=0.0,
        done=False,
        grader_score=None,
        metadata={
            "task": task,
            "noise_queries": 0,
            "redundant_queries": 0,

            # NEW: Tracks useful vs useless reasoning behavior
            "relevant_queries": 0,
            "critical_discoveries": 0,
            "safe_actions": 0,

            "document_verified": False,
        },
    )


# =========================================================
# MAIN ENVIRONMENT CLASS
# =========================================================

class SchemeEnvEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = False
    _shared_state = {}

    def __init__(self):
        super().__init__()

        if not SchemeEnvEnvironment._shared_state:
            persona = generate_dynamic_persona(1)
            obs = _make_fresh_obs(1, persona)
            state = State(episode_id=str(uuid4()), step_count=0)

            SchemeEnvEnvironment._shared_state = {
                "task": 1,
                "persona": persona,
                "state": state,
                "obs": obs,
            }

        self._load_shared()

    def _load_shared(self):
        s = SchemeEnvEnvironment._shared_state
        self._task = s["task"]
        self._persona = s["persona"]
        self._state = s["state"]
        self._obs = s["obs"]

    def _save_shared(self):
        SchemeEnvEnvironment._shared_state.update({
            "task": self._task,
            "persona": self._persona,
            "state": self._state,
            "obs": self._obs,
        })

    def reset(self, seed=None, **kwargs) -> Observation:
        self._task = seed if seed in (1, 2, 3, 4) else (self._task % 4) + 1
        self._persona = generate_dynamic_persona(self._task)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._obs = _make_fresh_obs(self._task, self._persona)
        self._save_shared()
        return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        obs = self._obs
        task = self._task
        persona = self._persona

        if action.action_type == "ask_question":
            key = (action.value or "").strip()

            if key in NOISE_FIELDS:
                obs.metadata["noise_queries"] += 1
                obs.reward = -0.7

            elif key in obs.known_profile:
                obs.metadata["redundant_queries"] += 1
                obs.reward = -0.5

            elif key in VALID_QUERY_FIELDS and key in persona:
                obs.known_profile[key] = persona[key]
                obs.metadata["relevant_queries"] += 1
                obs.reward = 0.5

        elif action.action_type == "request_document":
            if task == 4 and "pan" in (action.value or "").lower():
                obs.metadata["critical_discoveries"] += 1
                obs.metadata["document_verified"] = True
                obs.reward = 1.5
            else:
                obs.reward = 0.5

        elif action.action_type in ["approve_scheme", "reject_applicant", "escalate"]:

            base_outcome = "fail"

            if action.action_type == "approve_scheme":
                if action.value == persona.get("optimal_scheme"):
                    base_outcome = "optimal"
                elif action.value in persona.get("eligible_schemes", []):
                    base_outcome = "suboptimal"

            elif action.action_type == "reject_applicant":
                if task == 3:
                    base_outcome = "optimal"

            elif action.action_type == "escalate":
                if task == 4:
                    base_outcome = "safe"

            # FINAL REWARD CALCULATION (NEW SYSTEM)
            total_reward, final_score = _compute_final_reward(
                task, obs, self._state.step_count, persona, base_outcome
            )

            obs.reward = total_reward
            obs.grader_score = final_score
            obs.done = True
            obs.is_terminated = True

        return self._finalize_step(obs)

    def _finalize_step(self, obs: Observation) -> Observation:
        if self._state.step_count >= MAX_STEPS and not obs.done:
            obs.is_terminated = True
            obs.reward = -2.0
            obs.done = True
            obs.grader_score = 0.0

        self._obs = obs
        self._save_shared()
        return obs

    @property
    def state(self) -> State:
        return self._state
