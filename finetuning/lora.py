import torch


class LoRA(torch.nn.Module):
    def __init__(self, linear:torch.nn.Linear, r:int, alpha_is_equals:bool, bias:bool = False):
        super().__init__()

        self.linear = linear
        self.scale = r / (r if alpha_is_equals is True else 2 * r)
        self.A = torch.nn.Parameter(torch.randn(r, linear.in_features) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(linear.out_features, r))
        self.linear.weight.requires_grad = False

    def forward(self, x):
        linear = self.linear(x)
        dot_product = (x @ self.A.T) @ (self.B.T)
        return linear + dot_product