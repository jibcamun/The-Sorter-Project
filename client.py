from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SorterAction, SorterObservation, SorterState
except ImportError:
    from models import SorterAction, SorterObservation, SorterState


class SorterEnv(EnvClient[SorterAction, SorterObservation, SorterState]):

    @staticmethod
    def _observation_kwargs(payload: Dict) -> Dict:
        kwargs = {
            "grid_dims": payload.get("grid_dims"),
            "weighted_grid": payload.get("weighted_grid"),
            "current_grid": payload.get("current_grid"),
            "reward": payload.get("reward", ([], [])),
            "done": payload.get("done", False),
        }

        optional_fields = (
            "objects_present",
            "positions_place",
            "positions_segment",
            "positions_adjust",
            "positions",
        )
        for field in optional_fields:
            if field in payload:
                kwargs[field] = payload[field]

        return kwargs

    def _step_payload(self, action: SorterAction) -> Dict:

        return {
            "segment": action.segment,
            "adjust": action.adjust,
            "place": action.place,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SorterObservation]:

        obs_data = payload.get("observation", {})
        observation = SorterObservation(**self._observation_kwargs(obs_data))

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SorterState:
        return SorterState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            **self._observation_kwargs(payload),
        )
