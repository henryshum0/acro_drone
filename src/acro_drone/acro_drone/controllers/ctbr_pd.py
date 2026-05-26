"""Collective thrust and body-rate PD controller."""

from __future__ import annotations

import torch

from .ctbr_pd_cfg import CtbrPdControllerCfg


class CtbrPdController:
    """PD controller that outputs per-motor thrust from a CTBR target.

    The target and state are shaped as [num_envs, 4] with columns:
    [collective thrust, p, q, r].
    """

    def __init__(self, cfg: CtbrPdControllerCfg, num_envs: int, device: str | torch.device) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.kp = torch.tensor(cfg.kp, dtype=torch.float32, device=self.device)
        self.kd = torch.tensor(cfg.kd, dtype=torch.float32, device=self.device)
        self.allocation_matrix = torch.tensor(
            cfg.allocation_matrix, dtype=torch.float32, device=self.device
        )

        if self.kp.shape != (3,) or self.kd.shape != (3,):
            raise ValueError(f"Expected kp/kd shape (3,), got {self.kp.shape} / {self.kd.shape}.")
        if self.allocation_matrix.shape != (4, 4):
            raise ValueError(
                f"Allocation matrix must have shape (4, 4), got {self.allocation_matrix.shape}."
            )

        try:
            self.allocation_matrix_inv = torch.linalg.inv(self.allocation_matrix)
        except RuntimeError:
            self.allocation_matrix_inv = torch.linalg.pinv(self.allocation_matrix)

        self.thrust_min = cfg.thrust_min
        self.thrust_max = cfg.thrust_max
        self.dt = float(cfg.dt)
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}.")

        self._prev_error: torch.Tensor = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)

    def reset_idx(self, env_ids: list[int]) -> None:
        """Reset any controller state."""
        self._prev_error[env_ids] = torch.zeros_like(self._prev_error[env_ids])

    def compute(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """Compute motor thrusts from CTBR target and current state.

        Args:
            target: Desired [collective thrust, p, q, r] with shape [num_envs, 4].
            current: Current [collective thrust, p, q, r] with shape [num_envs, 4].

        Returns:
            Motor thrusts with shape [num_envs, 4].
        """

        target = target.to(device=self.device, dtype=torch.float32)
        current = current.to(device=self.device, dtype=torch.float32)
        if target.shape[-1] != 4 or current.shape[-1] != 4:
            raise ValueError(
                f"Target/current must have last dimension 4, got {target.shape} / {current.shape}."
            )

        error = target[:, 1:4] - current[:, 1:4]

        # print(error)

        error_d = (error - self._prev_error) / self.dt

        self._prev_error = error.detach()

        # Compute wrench from error, then map to motor thrusts.
        torque = self.kp * error + self.kd * error_d
        wrench = torch.cat((target[:, :1], torque), dim=1)
        thrusts = (self.allocation_matrix_inv @ wrench.T).T
        
        # Bias all motors so none fall below thrust_min (preserves torque).
        if self.thrust_min is not None:
            min_per_env = thrusts.min(dim=1, keepdim=True).values
            bias = torch.clamp(self.thrust_min - min_per_env, min=0.0)
            thrusts = thrusts + bias
        # print(thrusts)
        # print(thrusts.sum(dim=1))
        # print(target[:, :1].sum(dim=1))
        # input()
        if self.thrust_max is not None:
            thrusts = torch.clamp(thrusts, max=self.thrust_max)

        return thrusts
