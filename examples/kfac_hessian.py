import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F

def compute_layerwise_hessian(loss: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
    """Compute Hessian for a single layer's parameters"""
    dw = grad(loss, param,
              create_graph=True, # necessary for higher order gradients
             )[0].flatten()
    n = dw.numel();  H = torch.zeros(n, n)
    for i in range(n):
        d2w = grad(dw[i], param, retain_graph=True)[0].flatten()
        H[i] = d2w
    return H

def compute_fisher_block(activations: torch.Tensor, gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KFAC Fisher approximation components A, G"""
    batch_size = activations.size(0)

    # Compute activation covariance A
    a_flat = activations.view(batch_size, -1)
    A = (a_flat.t() @ a_flat) / batch_size

    # Compute gradient covariance G
    g_flat = gradients.view(batch_size, -1)
    G = (g_flat.t() @ g_flat) / batch_size

    return A, G

def compare_hessians():
    # Create small network and data
    model = nn.Sequential(nn.Linear(5, 3))
    batch_size = 4
    x = torch.randn(batch_size, 5)
    y = torch.randint(0, 3, (batch_size,))

    # Forward pass
    out = model(x)
    loss = F.cross_entropy(out, y)

    # Store activations and gradients for KFAC
    activations = x
    gradients = grad(loss, out, create_graph=True)[0]

    # Compute exact Hessian
    exact_H = compute_layerwise_hessian(loss, model[0].weight)

    # Compute KFAC approximation
    A, G = compute_fisher_block(activations, gradients)
    # Materialize full KFAC matrix for comparison (normally avoided)
    kfac_H = torch.kron(A, G)

    # Compare
    print("Exact Hessian:", exact_H.shape)
    print("KFAC approx:", kfac_H.shape)
    print("Frobenius norm of difference:", torch.norm(exact_H - kfac_H))

    print(exact_H[:3,:3])
    print(kfac_H[:3,:3])
    print((exact_H - kfac_H)[:3,:3])

if __name__ == "__main__":
    compare_hessians()