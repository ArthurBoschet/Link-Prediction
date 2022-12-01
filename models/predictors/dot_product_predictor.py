import torch
import torch.nn as nn

class DotProduct(torch.nn.Module):
  def __init__(self):
    super(DotProduct, self).__init__()
    self.linear = nn.Linear(1, 1, bias=True)

  def forward(self, nodes_first, nodes_second):
    
    #dot product computation between node vectors
    dot_product = torch.sum(nodes_first * nodes_second, dim=-1)

    #(1,1) linear layer with bias to potentially help performance
    x = self.linear(dot_product)

    return x