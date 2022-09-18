"""
STATUS: DEV

- better way to convert one hot to atom type

"""

import torch

from typing import Dict, List


def _one_hot_to_atom_type(one_hot: torch.Tensor, key: Dict[str, torch.Tensor]) -> List[str]:
    atom_types = []
    for atom_tensor in one_hot:
        for k, v in key.items():
            if atom_tensor == v:
                atom_types.append(k)
    return atom_types
