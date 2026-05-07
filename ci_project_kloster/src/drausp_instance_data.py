import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Sequence, List

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent


@dataclass
class DrauspInstanceData:
    capacity_vector: np.ndarray
    requests: np.ndarray
    revenues: np.ndarray
    num_dimensions: int
    num_stages: int
    num_moves_per_request: np.ndarray
    num_requests: int = field(init=False)
    num_moves: int = field(init=False)
    penalty_factor: float = 1.0

    def __post_init__(self):
        self.num_moves_per_request = np.asarray(
            self.num_moves_per_request, dtype=np.int32
        )
        self.num_requests = len(self.requests)
        if len(self.num_moves_per_request) != self.num_requests:
            raise ValueError(
                "num_moves_per_request must contain one entry per request."
            )
        self.num_moves = int(self.requests[0].shape[0])


def crop_zeros(vec: Sequence[int]) -> List[int]:
    return list(filter(lambda a: a != 0, vec))


def cropped_ucap(request: Sequence[int]) -> List[int]:
    cropped = crop_zeros(request)
    cropped = cropped[::-1]
    cropped = crop_zeros(cropped)
    return cropped[::-1]


def substitute_bundles(total_dims: int, ucap: Sequence[int]) -> List[List[int]]:
    ucap_len = len(ucap)
    bundles = []
    for i in range(total_dims - ucap_len + 1):
        bundle = [0] * total_dims
        bundle[i : i + len(ucap)] = ucap
        bundles.append(bundle)
    return bundles


def pool_with_substitutes(
    pool: np.ndarray, num_dimensions: int
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    num_moves_per_request = np.empty(pool.shape[0], dtype=np.int32)
    max_bundle_size = 0
    for request_idx, request in enumerate(pool):
        cropped = cropped_ucap(request)
        bundles = substitute_bundles(num_dimensions, cropped)
        num_moves_per_request[request_idx] = len(bundles)
        if len(bundles) > max_bundle_size:
            max_bundle_size = len(bundles)
        result.append(bundles)
    result_np = np.empty(
        (pool.shape[0], max_bundle_size, num_dimensions), dtype=np.int32
    )
    for i, bundles in enumerate(result):
        bundle_size = len(bundles)
        for j, request in enumerate(bundles):
            result_np[i, j] = request
        if bundle_size < max_bundle_size:
            result_np[i, bundle_size:max_bundle_size] = bundles[0]
    return result_np, num_moves_per_request


# For old instance file format
def parse_instance_name(
    instance_path: str,
    moves: bool | None,
    num_dimensions: int | None,
    dim_capacity: int | None,
    num_stages: int | None,
) -> tuple[bool | None, int | None, int | None, int | None]:
    match = re.search(
        r"(([SW])(.*)-K(\d*)-R(\d*)-N(\d*)-lambda(\d*)-C(\d*)-T(\d*))", instance_path
    )
    if match is not None:
        moves = moves if moves is not None else True if match.group(2) == "S" else False
        num_dimensions = (
            num_dimensions if num_dimensions is not None else int(match.group(4))
        )
        dim_capacity = dim_capacity if dim_capacity is not None else int(match.group(8))
        num_stages = num_stages if num_stages is not None else int(match.group(9))
    if (
        moves is None
        or num_dimensions is None
        or dim_capacity is None
        or num_stages is None
    ):
        raise Exception(
            "You have to set either moves, num_dimensions, dim_capacity, and num_stages manually or provide a file with a valid name."
        )
    return moves, num_dimensions, dim_capacity, num_stages


def get_instance_data(
    instance_path: Union[str, Path], num_stages: int, include_reject_moves: bool = False
) -> DrauspInstanceData:
    instance_path = Path(instance_path).expanduser()
    moves = instance_path.stem[0]
    if moves not in {"S", "W"}:
        raise ValueError(
            f"Expected instance file name to start with 'S' or 'W', got {instance_path.name!r}."
        )
    moves = True if moves == "S" else False

    with instance_path.open("r") as file:
        capacity_vector = np.array(file.readline().split(), dtype=np.int32)
        revenues = np.array(file.readline().split(), dtype=np.float32) * 100
        requests = np.loadtxt(file, dtype=np.int32)

    num_dimensions = capacity_vector.shape[0]

    if moves:
        requests, num_moves_per_request = pool_with_substitutes(
            requests, num_dimensions
        )
    else:
        requests = np.expand_dims(requests, 1)
        num_moves_per_request = np.ones(requests.shape[0], dtype=np.int32)

    if include_reject_moves:
        # add reject action as a reqeuest with demand 0
        extended_requests = np.zeros(
            (
                requests.shape[0],
                requests.shape[1] + 1,
                requests.shape[2],
            ),
            dtype=requests.dtype,
        )
        extended_requests[:, 1:, :] = requests
        requests = extended_requests
        num_moves_per_request = num_moves_per_request + 1
    return DrauspInstanceData(
        capacity_vector,
        requests,
        revenues,
        num_dimensions,
        num_stages,
        num_moves_per_request,
    )


def random_policy_rollout(
    instance_data: DrauspInstanceData,
    rng: np.random.Generator,
    num_episodes: int,
) -> np.ndarray:
    has_reject_move = np.all(instance_data.requests[:, 0, :] == 0)
    revenues = []

    for _ in range(num_episodes):
        capacity_vector = instance_data.capacity_vector.copy()
        revenue = 0.0
        rollout = rng.integers(
            low=0, high=instance_data.num_requests, size=instance_data.num_stages
        )

        for current_request_idx in rollout:
            current_requests = instance_data.requests[current_request_idx]
            valid_requests = np.all((capacity_vector - current_requests) >= 0, axis=1)
            valid_indices = np.flatnonzero(valid_requests)
            sampled_index = rng.choice(valid_indices)
            capacity_vector -= current_requests[sampled_index]
            if not (has_reject_move and sampled_index == 0):
                revenue += instance_data.revenues[current_request_idx]

        revenues.append(revenue)

    return np.array(revenues)


def random_policy_rollout_batched(
    instance_data: DrauspInstanceData,
    rng: np.random.Generator,
    num_episodes: int,
) -> np.ndarray:
    rollout = rng.integers(
        low=0,
        high=instance_data.num_requests,
        size=(num_episodes, instance_data.num_stages),
    )
    capacity_vectors = np.broadcast_to(
        instance_data.capacity_vector, (num_episodes, instance_data.num_dimensions)
    ).copy()
    revenues = np.zeros(num_episodes, dtype=instance_data.revenues.dtype)

    for stage_idx in range(instance_data.num_stages):
        current_request_idx = rollout[:, stage_idx]
        current_requests = instance_data.requests[current_request_idx]
        valid_requests = np.all(
            (capacity_vectors[:, np.newaxis, :] - current_requests) >= 0, axis=2
        )
        valid_counts = valid_requests.sum(axis=1, dtype=np.int32)
        sampled_ranks = rng.integers(low=0, high=valid_counts, size=num_episodes)
        sampled_index = np.argmax(
            np.cumsum(valid_requests, axis=1) > sampled_ranks[:, np.newaxis], axis=1
        )

        chosen_requests = current_requests[np.arange(num_episodes), sampled_index]
        capacity_vectors -= chosen_requests
        revenues += instance_data.revenues[current_request_idx] * (sampled_index != 0)

    return revenues


if __name__ == "__main__":
    instance_data = get_instance_data(
        PROJECT_ROOT / "instances" / "wendtris" / "S-wendtris12D.txt",
        20,
        include_reject_moves=True,
    )
    num_episodes = 100_000
    seed = 0

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    revenues = random_policy_rollout(instance_data, rng, num_episodes=num_episodes)
    runtime = time.perf_counter() - start_time
    print(f"random_policy_rollout mean revenue: {revenues.mean():.4f}")
    print(f"random_policy_rollout runtime: {runtime:.4f}s")

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    batched_revenues = random_policy_rollout_batched(
        instance_data, rng, num_episodes=num_episodes
    )
    runtime = time.perf_counter() - start_time
    print(f"random_policy_rollout_batched mean revenue: {batched_revenues.mean():.4f}")
    print(f"random_policy_rollout_batched runtime: {runtime:.4f}s")
