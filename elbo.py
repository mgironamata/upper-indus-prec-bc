from plum import dispatch
from varz.torch import Vars
import torch.nn as nn
import torch
from stheno.torch import B, GP, EQ, Normal

"""Acknowledgement: this class constructor was provided by Wessel Bruisma. Thanks!"""

# Detect device.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

__all__ = ['ApproximatePosterior']


class ApproximatePosterior(nn.Module):
    def __init__(self, dim: int, use_device = device):
        nn.Module.__init__(self)

        # Get learnable parameters.
        num_params = int(dim + (dim * dim + dim) / 2)
        self.source = nn.Parameter(
            torch.randn(num_params, device= use_device), requires_grad=True
        )
        self.dim = dim

    # Build normal distribution.
    def build_normal(self):
        vs = Vars(self.source.dtype, source=self.source)
        cov = vs.positive_definite(shape=(self.dim, self.dim), name="cov")
        mean = vs.unbounded(shape=(self.dim, 1), name="mean")
        self.dist = Normal(mean, cov)

        # Redirect methods to `dist`.
        for name in ["kl", "sample"]:  
            setattr(self, name, getattr(self.dist, name))
        

if __name__ == "__main__":

    # Let all of Stheno run on that device.
    B.device(device).__enter__()

    B.epsilon = 1e-6  # Needs to be relatively high for `float32`s.

    f = GP(EQ().stretch(0.2))
    x = torch.randn(1000, 2, device=device)
    x_ind = torch.randn(100, 2, device=device)

    # Compute ELBO (roughly).

    # Build approximate posterior:
    q = ApproximatePosterior(100).to(device)
    q.build_normal()

    # KL:
    kl = q.kl(f(x_ind))
    print(q.sample().shape)
    print(f(x_ind).sample())

    # Reconstruction term:
    f_post = f | (f(x_ind), q.sample())
    f_sample = f_post(x_ind).sample()
    recon = torch.sum(f_sample ** 2)  # Compute reconstruction term here.

    elbo = recon - kl
    (-elbo).backward()
    print(q.source.grad)
