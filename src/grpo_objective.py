"""Token-level clipped GRPO surrogate with a separate frozen-reference KL."""
import torch


def grpo_objective(current, behavior, reference, advantages, clip_eps=.2, kl_coef=.05):
    if not (len(current) == len(behavior) == len(reference) == len(advantages)):
        raise ValueError('One log-probability sequence and advantage per completion required')
    if not current:
        raise ValueError('Cannot optimize an empty branch group')
    losses = []
    for new, old, ref, advantage in zip(current, behavior, reference, advantages):
        if new.shape != old.shape or new.shape != ref.shape or new.numel() == 0:
            raise ValueError('Aligned nonempty completion tokens required')
        ratio = (new - old.detach()).exp()
        pg = -torch.minimum(ratio * advantage, ratio.clamp(1-clip_eps,1+clip_eps) * advantage)
        delta = ref.detach() - new
        kl = delta.exp() - delta - 1
        loss = (pg + kl_coef*kl).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite GRPO objective')
        losses.append(loss)
    return torch.stack(losses).mean()
