"""Scheme Env Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import SchemeAction, SchemeObservation


class SchemeEnv(EnvClient[SchemeAction, SchemeObservation, State]):
    """Client for the Scheme Env Environment."""

    def _step_payload(self, action: SchemeAction) -> Dict:
        return {
            "action_type": action.action_type,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SchemeObservation]:
        obs_data = payload.get("observation", {})
        observation = SchemeObservation(
            known_profile=obs_data.get("known_profile", {}),
            missing_data=obs_data.get("missing_data", []),
            notification=obs_data.get("notification", ""),
            is_terminated=obs_data.get("is_terminated", False),
            grader_score=obs_data.get("grader_score", None),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )