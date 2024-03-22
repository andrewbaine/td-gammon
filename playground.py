import torch
import torch.nn as nn

weights = torch.tensor([[-2, -1, 1, 2]], dtype=torch.float32)

linear_layer = nn.Linear(4, 1, bias=False)

linear_layer.weight = nn.Parameter(weights)

input_tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
output = linear_layer(input_tensor)
print(output)
print(output.item())
