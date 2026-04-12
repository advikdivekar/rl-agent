import json
import urllib.request
import urllib.error
from typing import Any

import gymnasium as gym
import numpy as np


# Base URL of the running Scheme Env FastAPI server.
# Reads from environment variable if set, otherwise defaults to localhost.
import os
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


def _post(path: str, body: dict) -> dict:
    # Send a JSON POST request to the environment server and return the response.
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ENV_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


class SchemeEnvGym(gym.Env):
    """
    Gymnasium-compatible wrapper around the Scheme Env HTTP server.

    Translates standard gym calls (reset, step) into HTTP requests
    to the existing /reset and /step endpoints. This makes the environment
    compatible with any RL training library that expects a gymnasium.Env,
    such as Stable Baselines3, RLlib, or custom PPO/GRPO loops.

    The environment server must be running before this wrapper is used.
    Start it with: uvicorn server.app:app --port 7860
    """

    # Metadata required by gymnasium
    metadata = {"render_modes": []}

    def __init__(self, task: int = 1):
        super().__init__()

        # Which task to run (1-5). Passed as seed to /reset.
        self.task = task

        # Current observation dict returned by the server.
        self._obs = {}

        # Action space: 5 discrete action types mapped to integers.
        # 0=ask_question, 1=request_document, 2=approve_scheme,
        # 3=reject_applicant, 4=escalate
        self.action_space = gym.spaces.Discrete(5)

        # Observation space: flat dict of text fields.
        # Using Text space since observations are structured JSON, not arrays.
        # Downstream agents are expected to read self.last_obs directly.
        self.observation_space = gym.spaces.Text(min_length=0, max_length=4096)

        # Maps integer action index to (action_type, default_value) pairs.
        # Agents that need specific values should call step_with_action() instead.
        self._action_map = {
            0: ("ask_question",      "income"),
            1: ("request_document",  "aadhaar_card"),
            2: ("approve_scheme",    "PMKVY"),
            3: ("reject_applicant",  "NO_ELIGIBLE_SCHEME"),
            4: ("escalate",          "MANUAL_REVIEW_REQUIRED"),
        }

        # Last raw observation dict — agents read this for full state detail.
        self.last_obs = {}

    def reset(self, seed=None, options=None):
        # Call /reset on the server with the configured task as the seed.
        result = _post("/reset", {"seed": self.task})
        self._obs = result.get("observation", result)
        self.last_obs = self._obs

        # Return observation as JSON string plus empty info dict.
        return json.dumps(self._obs), {}

    def step(self, action: int):
        # Map the integer action to an action_type and default value.
        action_type, value = self._action_map[action]
        return self.step_with_action(action_type, value)

    def step_with_action(self, action_type: str, value: str):
        """
        Execute a named action directly instead of using the integer mapping.
        Use this when the agent needs to pass a specific value, for example:
            env.step_with_action("approve_scheme", "PMAY")
            env.step_with_action("ask_question", "occupation")
        """
        result = _post("/step", {"action": {"action_type": action_type, "value": value}})

        obs     = result.get("observation", result)
        reward  = float(result.get("reward", 0.0))
        done    = bool(result.get("done", False))

        # terminated = episode ended by environment decision (correct action or wrong action)
        # truncated  = episode ended by step limit (timeout notification)
        # FIX: original logic compared is_terminated (bool) to string "timeout" which
        # was always False, making terminated always equal to done. Now both flags
        # are derived from the notification string which is the authoritative signal.
        terminated = done and not obs.get("notification", "").startswith("TIMEOUT")
        truncated  = done and obs.get("notification", "").startswith("TIMEOUT")

        self.last_obs = obs

        # Gymnasium expects (obs, reward, terminated, truncated, info)
        return json.dumps(obs), reward, terminated, truncated, {"grader_score": obs.get("grader_score")}

    def render(self):
        # Print current state to stdout for debugging.
        print(json.dumps(self.last_obs, indent=2))