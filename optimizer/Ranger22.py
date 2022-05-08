# first draft for Ranger22
# removes scheduler into it's own class along with various improvements (thanks to @AdrienCourtois)
# pending - normalized gradient descent where norm is calced layer by layer (idea credit AdrienCourtois)

from typing import Tuple
import torch
import torch.nn.functional as F
import math
import collections


def unit_norm(x):
    """axis-based Euclidean norm"""
    # PARTIALLY REWORKED
    # We need to know exactly what the paper meant by that.
    # Not handled: linear with heads (is that even a thing?)

    if x.ndim == 1:
        return x.norm()

    dims = tuple(range(1, x.ndim))
    return x.norm(dim=dims, p=2, keepdim=True)


def adaptive_grad_clipping(p, alpha=1e-2, eps=1e-3):
    param_norm = unit_norm(p).clamp(min=eps)
    grad_norm = unit_norm(p.grad)

    max_grad_norm = alpha * param_norm
    clipped_grad = p.grad * max_grad_norm / grad_norm.clamp(min=1e-6)

    new_grads = torch.where(grad_norm > max_grad_norm, clipped_grad, p.grad)
    p.grad.data.copy_(new_grads)


def centralize_gradient(x, gc_conv_only=False):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization"""

    size = x.dim()

    if gc_conv_only and size > 3:
        x.data.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    elif not gc_conv_only and size > 1:
        x.data.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))


class Ranger22(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 5e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        pos_neg_momentum: float = 1.0,
        weight_decay: float = 1e-4,
        eps: float = 1e-8,
        normloss: bool = True,
        stable_weight_decay: bool = False,
        amsgrad: bool = False,
        adabelief: bool = False,
        softplus: bool = True,
        beta_softplus: int = 50,
        grad_central: bool = True,
        gc_conv_only: bool = False,
        ada_grad_clip: bool = True,
        agc_clip_val: float = 1e-2,
        agc_eps: float = 1e-3,
        lookahead_active: bool = True,
        la_mergetime: int = 5,
        la_alpha: float = 0.5,
    ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            normloss=normloss,
            stable_weight_decay=stable_weight_decay,
            amsgrad=amsgrad,
            pos_neg_momentum=pos_neg_momentum,
            grad_central=grad_central,
            gc_conv_only=gc_conv_only,
            adabelief=adabelief,
            ada_grad_clip=ada_grad_clip,
            agc_clip_val=agc_clip_val,
            agc_eps=agc_eps,
            softplus=softplus,
            beta_softplus=beta_softplus,
            lookahead_active=lookahead_active,
            la_mergetime=la_mergetime,
            la_alpha=la_alpha,
            la_step=0,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("la_step", 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None and isinstance(closure, collections.Callable):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            pos_neg_momentum = group["pos_neg_momentum"]
            normloss = group["normloss"]
            stable_weight_decay = group["stable_weight_decay"]
            beta1, beta2 = group["betas"]

            param_size = 0
            variance_ma_sum = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_size += p.numel()

                if p.grad.is_sparse:
                    raise RuntimeError("sparse matrix not supported atm")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    state["grad_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["variance_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    if group["lookahead_active"]:
                        state["lookahead_params"] = torch.zeros_like(p.data)
                        state["lookahead_params"].copy_(p.data)

                    if group["pos_neg_momentum"]:
                        state["neg_grad_ma"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    if group["amsgrad"]:
                        state["max_variance_ma"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                # Gradient clipping
                if group["ada_grad_clip"]:
                    adaptive_grad_clipping(
                        p, alpha=group["agc_clip_val"], eps=group["agc_eps"]
                    )

                # Gradient centralization
                if group["grad_central"]:
                    centralize_gradient(p.grad, gc_conv_only=group["gc_conv_only"])

                # TODO: Gradient normalization

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                variance_ma = state["variance_ma"]

                # Update the running mean and variance of the gradients
                if group["pos_neg_momentum"]:
                    if state["step"] % 2 == 1:
                        grad_ma = state["grad_ma"]
                    else:
                        grad_ma = state["neg_grad_ma"]

                    grad_ma.mul_(beta1**2).add_(p.grad, alpha=1 - beta1**2)
                else:
                    grad_ma = state["grad_ma"]
                    grad_ma.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                if group["adabelief"]:
                    grad_residual = p.grad - grad_ma
                    variance_ma.mul_(beta2).addcmul_(
                        grad_residual, grad_residual, value=1 - beta2
                    )
                else:
                    variance_ma.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Compute the average variance across all the parameters for stable weight decay
                if not normloss and weight_decay and stable_weight_decay:
                    variance_ma_debiased = variance_ma / bias_correction2
                    variance_ma_sum += variance_ma_debiased.sum()

            if not normloss and weight_decay and stable_weight_decay:
                variance_normalized = torch.sqrt(variance_ma_sum / param_size)

                if torch.any(variance_normalized != variance_normalized):
                    raise RuntimeError("hit nan for variance_normalized")

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr / bias_correction1

                # (Stable) weight decay, NormLoss
                if weight_decay:
                    if normloss:
                        unorm = unit_norm(p.data)
                        correction = 2 * weight_decay * (1 - torch.div(1, unorm + eps))
                        p.data.mul_(1 - lr * correction)

                    elif stable_weight_decay:
                        correction = weight_decay / variance_normalized
                        p.data.mul_(1 - lr * correction)

                    else:
                        correction = weight_decay
                        p.data.mul_(1 - lr * correction)

                # Fetching mean and variance
                variance_ma = state["variance_ma"]

                if pos_neg_momentum:
                    if state["step"] % 2 == 1:
                        grad_ma, neg_grad_ma = (
                            state["grad_ma"],
                            state["neg_grad_ma"],
                        )
                    else:
                        grad_ma, neg_grad_ma = (
                            state["neg_grad_ma"],
                            state["grad_ma"],
                        )
                else:
                    grad_ma = state["grad_ma"]

                # Computing the denominator
                if group["amsgrad"]:
                    max_variance_ma = state["max_variance_ma"]

                    torch.max(max_variance_ma, variance_ma, out=max_variance_ma)
                    denom = max_variance_ma.sqrt() / math.sqrt(bias_correction2)
                else:
                    if group["adabelief"]:
                        denom = variance_ma.sqrt() / math.sqrt(bias_correction2)
                    else:
                        denom = variance_ma.sqrt() / math.sqrt(bias_correction2)

                if group["softplus"]:
                    denom = F.softplus(denom, beta=group["beta_softplus"])
                else:
                    denom.add_(eps)

                # Update the parameter
                if pos_neg_momentum:
                    noise_norm = math.sqrt(
                        (1 + pos_neg_momentum) ** 2 + pos_neg_momentum**2
                    )

                    pnmomentum = (
                        grad_ma.mul(1 + pos_neg_momentum)
                        .add(neg_grad_ma, alpha=-pos_neg_momentum)
                        .mul(1 / noise_norm)
                    )

                    p.addcdiv_(pnmomentum, denom, value=-step_size)

                else:
                    p.addcdiv_(grad_ma, denom, value=-step_size)

        # lookahead
        # ---------------------
        for group in self.param_groups:
            if not group["lookahead_active"]:
                continue

            group["la_step"] += 1
            la_alpha = group["la_alpha"]

            if group["la_step"] >= group["la_mergetime"]:
                group["la_step"] = 0

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(la_alpha).add_(
                        param_state["lookahead_params"],
                        alpha=1.0 - la_alpha,
                    )
                    param_state["lookahead_params"].copy_(p.data)

        return loss
