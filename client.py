from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SorterAction, SorterObservation, SorterState
except ImportError:
    from models import SorterAction, SorterObservation, SorterState


class SorterEnv(EnvClient[SorterAction, SorterObservation, SorterState]):

    def _step_payload(self, action: SorterAction) -> Dict:

        return {
            "segment": action.segment,
            "adjust": action.adjust,
            "place": action.place,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SorterObservation]:

        obs_data = payload.get("observation", {})
        observation = SorterObservation(
            grid_dims=obs_data.get("grid_dims"),
            weighted_grid=obs_data.get("weighted_grid"),
            current_grid=obs_data.get("current_grid"),
            positions_place=obs_data.get("positions_place", {}),
            positions_segment=obs_data.get("positions_segment", {}),
            positions_adjust=obs_data.get("positions_adjust", {}),
            reward=obs_data.get("reward", ([], [])),
            positions=obs_data.get("positions", []),
            done=obs_data.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SorterState:

        return SorterState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            grid_dims=payload.get("grid_dims"),
            weighted_grid=payload.get("weighted_grid"),
            objects_present=payload.get("objects_present", {}),
            current_grid=payload.get("current_grid"),
            positions_place=payload.get("positions_place", {}),
            positions_segment=payload.get("positions_segment", {}),
            positions_adjust=payload.get("positions_adjust", {}),
            reward=payload.get("reward", ([], [])),
            done=payload.get("done", False),
        )
