import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Sequence, List

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent


@dataclass
class DrauspInstanceData:
    num_slots: int
    pool_size: int  # Anzahl Kandidaten-Anfragen im Pool (|N| im Paper) - NICHT T_d!
    capacity_vector: list[int]
    instance: list
    request_length: int

def get_instance_data(data_path: Union[str, Path]):
    with data_path.open("r") as file:
        capacity_vector = np.array(file.readline().split(), dtype=int).tolist()
        revenues = np.array(file.readline().split(), dtype=np.float32)
        requests = np.loadtxt(file, dtype=np.int32)

    num_slots = len(capacity_vector)
    pool_size = len(requests)
    request_length = requests.shape[1]
    instance = []
    for i, request in enumerate(requests):
        row = [revenues[i]] + [int(x) for x in request] + [0] * (num_slots - len(request))
        instance.append(row)
    #print(instance)
    return(DrauspInstanceData(num_slots, pool_size, capacity_vector, instance, request_length))


if __name__ == "__main__":
    data = get_instance_data(PROJECT_ROOT / "instances" / "lion18s" / "SA01.txt")
    print(data.capacity_vector)
    print(data.request_length)