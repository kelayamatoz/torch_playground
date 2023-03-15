import torch
import torch.nn as nn
from torch._functorch.compilers import memory_efficient_fusion


class LogReg(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        return self.loss(self.linear(x), label)


def compiler_fn(fx_graph: torch.fx.GraphModule, *args, **kwargs):
    print(fx_graph.code)
    return fx_graph


if __name__ == '__main__':
    batch_dim = 2
    input_dim = 2
    output_dim = 2
    x = torch.ones((batch_dim, input_dim))
    y = torch.ones((batch_dim, output_dim))
    LogRegModule = LogReg(input_dim, output_dim)
    cloned_inputs = [x.clone().detach().requires_grad_(True)]
    AOTLogRegModule = memory_efficient_fusion(
        LogRegModule,
        fw_compiler=compiler_fn,
        bw_compiler=compiler_fn)
    loss = AOTLogRegModule(x, y)
    loss.backward()
