import os
from typing import BinaryIO, IO

import torch.nn


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    model_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()

    os.mkdir(out)
    torch.save(model_dict, os.path.join(out, f"model.pt"))
    torch.save(optimizer_dict, os.path.join(out, f"optimizer.pt"))
    torch.save({"iteration": iteration}, os.path.join(out, "config.json"))


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    model_dict = torch.load(os.path.join(src, f"model.pt"), weights_only=False)
    optimizer_dict = torch.load(os.path.join(src, f"optimizer.pt"), weights_only=False)
    config_dict = torch.load(os.path.join(src, f"config.json"))

    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)

    assert "iteration" in config_dict.keys()
    iteration = int(config_dict["iteration"])
    return iteration
