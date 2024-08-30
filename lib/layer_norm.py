import torch

class LayerNorm(torch.nn.Module):

	def __init__(self, n_embed, rotate=(0,2,1), eps=1e-6):
		super().__init__()
		self.layer_norm = torch.nn.LayerNorm(n_embed, eps=eps)
		self.rotate = rotate

	def forward(self, index):
		index = torch.permute(index, self.rotate)
		index = self.layer_norm(index)
		index = torch.permute(index, self.rotate)
		return index