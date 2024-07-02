import torch
import torch.nn.functional as F

# Example probabilities (after softmax)
probabilities = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.3, 0.4, 0.3, 0.1, 0.2, 0.9], [1.0, 1e-10, 1.0, 1.0, 1e-10, 1.0]])

# Compute entropy
entropy = -(probabilities * torch.log(probabilities)).sum(dim=1)

print("Information Entropy:", entropy)