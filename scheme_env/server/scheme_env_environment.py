from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Action, Observation


MAX_STEPS = 12

TASK1_PERSONA = {
    "age": "26", "income": "4000", "occupation": "mason",
    "has_aadhaar": "True", "optimal_scheme": "PMKVY",
    "eligible_schemes": ["PMKVY", "PMAY"],
}
TASK2_PERSONA = {
    "age": "35", "income": "3000", "occupation": "farm_labourer",
    "has_aadhaar": "True", "optimal_scheme": "MGNREGS",
    "eligible_schemes": ["MGNREGS"],
    "missing_keys": ["has_aadhaar", "occupation"],
}
TASK3_PERSONA = {
    "age": "14", "income": "500000", "occupation": "student",
    "has_aadhaar": "False", "optimal_scheme": None,
    "eligible_schemes": [],
}
PERSONAS = {1: TASK1_PERSONA, 2: TASK2_PERSONA, 3: TASK3_PERSONA}


class SchemeEnvEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task = 1
        self._persona = TASK1_PERSONA
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._obs = self._make_obs()

    def _make_obs(self) -> Observation:
        if self._task == 1:
            return Observation(
                known_profile={
                    "age": "26", "income": "4000",
                    "occupation": "mason", "has_aadhaar": "True",
                },
                missing_data=[],
                notification=(
                    "[TASK 1 / 3 - SCHEME DISCOVERY (Easy)] "
                    "Profile is COMPLETE. No missing data. "
                    "Identify and approve the optimal welfare scheme."
                ),
                is_terminated=False, reward=0.0, done=False,
                metadata={"task": 1},  # ← store task in metadata
            )
        elif self._task == 2:
            return Observation(
                known_profile={"age": "35", "income": "3000"},
                missing_data=["has_aadhaar", "occupation"],
                notification=(
                    "[TASK 2 / 3 - MISSING DATA (Medium)] "
                    "Profile is INCOMPLETE. Gather ALL missing data "
                    "before approving any scheme."
                ),
                is_terminated=False, reward=0.0, done=False,
                metadata={"task": 2},  # ← store task in metadata
            )
        elif self._task == 3:
            return Observation(
                known_profile={
                    "age": "14", "income": "500000",
                    "occupation": "student", "has_aadhaar": "False",
                },
                missing_data=[],
                notification=(
                    "[TASK 3 / 3 - CONFLICT RESOLUTION (Hard)] "
                    "Profile is COMPLETE. Age is 14, income is 500000. "
                    "This applicant does NOT qualify for any scheme. "
                    "You MUST reject this applicant."
                ),
                is_terminated=False, reward=0.0, done=False,
                metadata={"task": 3},  # ← store task in metadata
            )

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        if seed is not None and seed in (1, 2, 3):
            self._task = seed
        else:
            self._task = (self._task % 3) + 1

        self._persona = PERSONAS[self._task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._obs = self._make_obs()
        print(f"[ENV] Reset -> Task {self._task}, profile={self._obs.known_profile}")
        return self._obs

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        obs = self._obs

        # ← Read task from metadata (survives HTTP statelessness)
        current_task = obs.metadata.get("task", self._task)

        valid_actions = {
            "ask_question", "request_document",
            "approve_scheme", "reject_applicant", "escalate"
        }

        if action.action_type not in valid_actions:
            obs.notification = (
                f"Unknown action '{action.action_type}'. "
                f"Valid: {', '.join(valid_actions)}."
            )
            obs.reward = -1.0
            obs.done = False
            return self._check_timeout(obs)

        if action.action_type == "ask_question":
            key = action.value
            if key in obs.known_profile:
                obs.notification = f"'{key}' already known: {obs.known_profile[key]}. Redundant question."
                obs.reward = -1.0
            elif key in self._persona and key not in ("optimal_scheme", "eligible_schemes", "missing_keys"):
                val = self._persona[key]
                obs.known_profile[key] = val
                if key in obs.missing_data:
                    obs.missing_data.remove(key)
                obs.notification = f"Applicant answered: My {key} is {val}."
                obs.reward = 1.0
            else:
                obs.notification = f"'{key}' is not relevant to this case."
                obs.reward = -1.0

        elif action.action_type == "request_document":
            obs.notification = f"Document '{action.value}' received and verified."
            obs.reward = 0.5

        elif action.action_type == "approve_scheme":
            scheme = action.value

            if current_task == 3:
                obs.notification = (
                    "SAFETY VIOLATION: This applicant is ineligible for any scheme. "
                    "You must reject, not approve."
                )
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0
                return obs

            if current_task == 2 and len(obs.missing_data) > 0:
                obs.notification = (
                    f"PREMATURE APPROVAL: Missing data {obs.missing_data} not yet collected."
                )
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0
                return obs

            optimal = self._persona["optimal_scheme"]
            eligible = self._persona["eligible_schemes"]

            if scheme == optimal:
                obs.notification = f"SUCCESS: Applicant correctly enrolled in {scheme}."
                obs.reward = 10.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 1.0
            elif scheme in eligible:
                obs.notification = f"Enrolled in {scheme}, but {optimal} was more optimal."
                obs.reward = 3.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.5
            else:
                obs.notification = f"ERROR: Applicant is NOT eligible for {scheme}."
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0

        elif action.action_type == "reject_applicant":
            if current_task == 3:
                obs.notification = (
                    f"CORRECT: Applicant rejected. Reason: {action.value}. "
                    "Age 14 and income 500000 disqualify all schemes."
                )
                obs.reward = 5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 1.0
            else:
                obs.notification = "ERROR: This applicant IS eligible. Rejection is invalid."
                obs.reward = -5.0
                obs.done = True
                obs.is_terminated = True
                obs.metadata["grader_score"] = 0.0

        elif action.action_type == "escalate":
            obs.notification = "Case escalated to senior officer. Episode ends."
            obs.reward = -2.0
            obs.done = True
            obs.is_terminated = True
            obs.metadata["grader_score"] = 0.0

        return self._check_timeout(obs)

    def _check_timeout(self, obs: Observation) -> Observation:
        if self._state.step_count >= MAX_STEPS and not obs.done:
            obs.is_terminated = True
            obs.notification = "Maximum steps (12) reached. Case closed."
            obs.reward = -2.0
            obs.done = True
            obs.metadata["grader_score"] = 0.0
        return obs

    @property
    def state(self) -> State:
        return self._state