import torch
from torch.types import Number


@torch.no_grad()
def amp_to_db(
    x: torch.Tensor, eps=torch.finfo(torch.float64).eps, top_db=40
) -> torch.Tensor:
    x_db = 20 * torch.log10(x.abs() + eps)
    return torch.max(x_db, (x_db.max(-1).values - top_db).unsqueeze(-1))


@torch.no_grad()
def temperature_sigmoid(x: torch.Tensor, x0: float, temp_coeff: float) -> torch.Tensor:
    return torch.sigmoid((x - x0) / temp_coeff)


@torch.no_grad()
def linspace(
    start: Number, stop: Number, num: int = 50, endpoint: bool = True, **kwargs
) -> torch.Tensor:
    if endpoint:
        return torch.linspace(start, stop, num, **kwargs)
    else:
        return torch.linspace(start, stop, num + 1, **kwargs)[:-1]
