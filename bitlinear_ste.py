import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        gamma = w.abs().mean().clamp(min=1e-5)
        
        # 2. Scale and round to {-1, 0, 1}
        w_scaled = w / gamma
        w_q = w_scaled.round().clamp(-1, 1)
        
        ctx.save_for_backward(w, gamma)
        return w_q, gamma

    @staticmethod
    def backward(ctx, grad_output, grad_gamma=None):
        w, gamma = ctx.saved_tensors
        return grad_output


def quantize_activation_int8(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-127, 127)
    return x_q, scale


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features)**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x_q, x_scale = quantize_activation_int8(x
        w_q, w_scale = TernaryQuantizeSTE.apply(self.weight)
        out = F.linear(x_q.float(), w_q.float(), self.bias)
        out = out / (x_scale * w_scale)
        return out