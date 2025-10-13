import torch
import numpy as np
import numpy.typing as npt

prev_sample_idx = 0


def load_inputs_target_from_np_dataset(x: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_tokens, targets = torch.zeros(size=(batch_size, context_length)), torch.zeros(
        size=(batch_size, context_length))

    for b in range(batch_size):
        sample_idx = torch.randint(0, int(x.shape[0]) - context_length, size=(1,)).item()

        input_tokens[b, :] = torch.from_numpy(x[sample_idx: sample_idx + context_length].astype(np.int64))
        targets[b, :] = torch.from_numpy(x[sample_idx + 1: sample_idx + context_length + 1].astype(np.int64))

    input_tokens = input_tokens.to(device)
    targets = targets.to(device)
    return input_tokens, targets
