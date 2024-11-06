# lr scheduler class extracted from Ranger21 - credit @AdrienCourtois

import math
from typing import Optional, Union


class Scheduler22:
    def __init__(
        self,
        optimizer,
        num_batches_per_epoch: int,
        num_epochs: int,
        # Warmup
        use_warmup: bool = True,
        warmup_size: Optional[Union[float, int]] = None,
        warmup_type: str = "linear",
        # Warmdown
        use_warmdown: bool = True,
        warmdown_start_size: Union[float, int] = 0.72,
        warmdown_min_lr: float = 3e-5,
        warmdown_type: str = "linear",
    ):

        self.n_step = 0

        self.training_niter = num_epochs * num_batches_per_epoch
        self.optimizer = optimizer

        self.base_lr = [group["lr"] for group in optimizer.param_groups]

        # Warmup
        self.use_warmup = use_warmup
        self.warmup_type = warmup_type

        if self.use_warmup:
            self.warmup_niter = []

            if warmup_size is None:
                for group in optimizer.param_groups:
                    euristic_niter = math.ceil(2 / (1 - group["betas"][1]))

                    if euristic_niter > 0.45 * self.training_niter:
                        self.warmup_niter.append(int(0.22 * self.training_niter))
                    else:
                        self.warmup_niter.append(euristic_niter)

            elif type(warmup_size) == float:
                self.warmup_niter = [
                    int(warmup_size * self.training_niter)
                    for _ in optimizer.param_groups
                ]

            else:
                self.warmup_niter = [warmup_size for _ in optimizer.param_groups]

        # Warmdown
        self.use_warmdown = use_warmdown
        self.warmdown_type = warmdown_type
        self.warmdown_min_lr = warmdown_min_lr

        if self.use_warmdown:
            self.start_warmdown = [
                int(warmdown_start_size * self.training_niter)
                for _ in optimizer.param_groups
            ]

    def warmup(self):
        if not self.use_warmup:
            return

        for group_idx, group in enumerate(self.optimizer.param_groups):
            niter = self.warmup_niter[group_idx]
            lr = self.base_lr[group_idx]

            if self.n_step < niter:
                alpha = min(1.0, self.n_step / niter)

                if self.warmup_type == "linear":
                    group["lr"] = alpha * lr

                elif self.warmup_type == "loglinear":
                    group["lr"] = 10 ** (math.log10(lr) - 3 * (1 - alpha))

                else:
                    raise Exception(f"Warmup type `{self.warmup_type}` is not defined.")

            elif self.n_step == niter:
                group["lr"] = self.base_lr[group_idx]

    def warmdown(self):
        if not self.use_warmdown:
            return

        for group_idx, group in enumerate(self.optimizer.param_groups):
            warmdown_niter = self.start_warmdown[group_idx]

            if self.n_step <= warmdown_niter:
                continue

            lr = self.base_lr[group_idx]
            min_lr = self.warmdown_min_lr

            current_iter = self.n_step + 1 - warmdown_niter
            alpha = current_iter / (self.training_niter - warmdown_niter + 1)

            if self.warmdown_type == "linear":
                group["lr"] = max(min_lr, (1 - alpha) * lr + alpha * min_lr)

            elif self.warmdown_type == "loglinear":
                group["lr"] = 10 ** (
                    (1 - alpha) * math.log10(lr) + alpha * math.log10(min_lr)
                )

            elif self.warmdown_type == "cosine":
                ratio = min_lr / lr
                group["lr"] = lr * (
                    ratio + (1 - ratio) * (1 + math.cos(math.pi * alpha)) / 2
                )

            else:
                raise Exception(f"Warmdown type `{self.warmdown_type}` is not defined.")

    def step(self):
        self.n_step += 1

        self.warmup()
        self.warmdown()
