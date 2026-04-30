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
    num_slots: int
    num_requests: int
    capacity_vector: list[int]
    instance: list

def get_instance_data(data_path: Union[str, Path]):
    with data_path.open("r") as file:
        capacity_vector = np.array(file.readline().split(), dtype=int).tolist()
        revenues = np.array(file.readline().split(), dtype=np.float32) * 100
        requests = np.loadtxt(file, dtype=np.int32)
        
    num_slots = len(capacity_vector)
    num_requests = len(requests)
    instance = []
    for i, request in enumerate(requests):
        row = [int(revenues[i])] + [int(x) for x in request] + [0] * (num_slots - len(request))
        instance.append(row)
    #print(instance)
    return(DrauspInstanceData(num_slots, num_requests, capacity_vector, instance))

data_DRAUSP = get_instance_data(PROJECT_ROOT / "instances" / "lion18s" / "SA01.txt")
print(data_DRAUSP.capacity_vector)