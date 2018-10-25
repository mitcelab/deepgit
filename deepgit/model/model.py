import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, hidden_dim):
		super(Encoder, self).__init__()
		self.embed = nn.Embedding(num_embeddings, embedding_dim, max_norm=1)
		self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
		self.linear1 = nn.Linear(embedding_dim*2, hidden_dim)
		self.linear2 = nn.Linear(embedding_dim*2, hidden_dim)

	def forward(self, x, toggle=True):
		x = self.embed(x)
		h, _ = self.rnn(x)
		h = torch.mean(F.selu(h), dim=1)
		h = torch.cat([torch.mean(x, dim=1), h], dim=1)
		if toggle:
			return self.linear1(h)
		return self.linear2(h)