import torch
from torch.autograd import grad
import numpy as np

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

def compute_full_hessian(loss: torch.Tensor, params: list) -> torch.Tensor:
    """Compute full Hessian matrix across all parameters"""
    dw = grad(loss, params, create_graph=True)
    dw = torch.cat([g.flatten() for g in dw])
    n = dw.numel(); H = torch.zeros(n, n)
    for i in range(n):
        d2ws = []
        for p in params:
            d2w = grad(dw[i], p, retain_graph=True)[0]
            d2ws.append(d2w.flatten())
        H[i] = torch.cat(d2ws)
    return H

# Test code
model = torch.nn.Sequential(
    torch.nn.Linear(8, 4, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1, bias=False)
)

x = torch.randn(21, 8)
y = torch.randn(21, 1)
output = model(x)
loss = torch.nn.MSELoss()(output, y)

# Get layerwise Hessians
H1 = compute_layerwise_hessian(loss, model[0].weight)
H2 = compute_layerwise_hessian(loss, model[2].weight)

# Get full Hessian
H_full = compute_full_hessian(loss, list(model.parameters()))

# Extract blocks from full Hessian
block1_size = model[0].weight.numel()
block2_size = model[2].weight.numel()
block1 = H_full[:block1_size, :block1_size]  # First layer is 2x3 = block1_size parameters
block2 = H_full[block1_size:, block1_size:]  # Second layer is 3x1 = 3 parameters

# Compare
print("First layer max difference:", torch.max(torch.abs(H1 - block1)))
print("Second layer max difference:", torch.max(torch.abs(H2 - block2)))
print(H_full)