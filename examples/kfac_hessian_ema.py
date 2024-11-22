import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F

class LinearWithPreAct(nn.Linear):
    def forward(self, x):
        self.a = x.clone().detach() ## == a_{\ell-1}
        self.s = F.linear(x, self.weight, self.bias) # == s_{\ell}
        return self.s

class FisherEstimator:
    def __init__(self, model, lambda_ema=0.95):
        self.lambda_ema = lambda_ema
        self.A_ema = [0. for l in model if isinstance(l, LinearWithPreAct)]
        self.G_ema = [0. for l in model if isinstance(l, LinearWithPreAct)]
        self.iters = 0

    def compute_layerwise_kfac_hessian(self):
        res = []
        for _i, (A, G) in enumerate(zip(self.A_ema, self.G_ema)):
            res.append(torch.kron(A, G))
        return res

    def update_model(self, loss, model):
        lin_layers = [l for l in model if isinstance(l, LinearWithPreAct)]
        self.iters += 1
        for i, layer in enumerate(lin_layers):
            a = layer.a # a_{\ell-1}
            g = grad(loss, layer.s, create_graph=True)[0] # g_{\ell} = dL/da_{\ell}

            batch_size = a.size(0)

            # Compute current batch statistics
            a_flat = a.view(batch_size, -1)
            A_batch = (a_flat.t() @ a_flat) #/ batch_size

            g_flat = g.view(batch_size, -1)
            G_batch = (g_flat.t() @ g_flat) #/ batch_size

            # Update EMAs
            fac = min(1 - 1/self.iters, self.lambda_ema)
            self.A_ema[i] = fac * self.A_ema[i] + (1 - fac) * A_batch
            self.G_ema[i] = fac * self.G_ema[i] + (1 - fac) * G_batch

def compute_layerwise_hessian(loss: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
    dw = grad(loss, param, create_graph=True)[0].flatten()
    n = dw.numel()
    H = torch.zeros(n, n)
    for i in range(n):
        d2w = grad(dw[i], param, retain_graph=True)[0].flatten()
        H[i] = d2w
    return H


def sample_from_logits(logits, nsamp=10):
# def sample_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=nsamp, replacement=True).permute(1, 0) # (batch_size, nsamp)

def compare_hessians(num_batches=100):
    nin = 2; nhidden = 2; nout = 3
    model = nn.Sequential(
        LinearWithPreAct(nin, nhidden, bias=False),
        nn.ReLU(),
        LinearWithPreAct(nhidden, nout, bias=False),
    )
    fisher_estimator = FisherEstimator(model)

    for _ in range(num_batches):
        batch_size = 100
        x = torch.randn(batch_size, nin)
        # y = torch.randn(batch_size, nout)

        out = model(x)
        # ysamp = out + torch.randn(*out.shape)

        nsamp = 10
        ys = sample_from_logits(out, nsamp=nsamp) ## sample ys from the model!
        out = out.unsqueeze(0).repeat(nsamp, 1, 1) # (nsamp, batch_size, nout)

        out = out.view(-1, nout) # (nsamp*batch_size, nout)
        ys = ys.reshape(-1)         # (nsamp*batch_size)

        loss = F.cross_entropy(out, ys)

        fisher_estimator.update_model(loss, model)

    exact_H0 = compute_layerwise_hessian(loss, model[0].weight)
    exact_H2 = compute_layerwise_hessian(loss, model[2].weight)

    kfac_Hs = fisher_estimator.compute_layerwise_kfac_hessian()

    ndec = 4
    print("===layer 0===")
    print(torch.round(exact_H0, decimals=ndec))
    print(torch.round(kfac_Hs[0], decimals=ndec))

    print("===layer 1===")
    print(torch.round(exact_H2, decimals=ndec))
    print(torch.round(kfac_Hs[1], decimals=ndec))

if __name__ == "__main__":
    compare_hessians()