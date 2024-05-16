import json

from function_calling_sft import generate_trajectories
from function_calling_sft.trajectories import trajectory_from_dict


def test_jsonl_payload_round_trip_revalidates_calls_and_tool_links() -> None:
    trajectory = generate_trajectories(2, seed=9)[0]
    restored = trajectory_from_dict(json.loads(json.dumps(trajectory.to_dict())))
    assert restored == trajectory
