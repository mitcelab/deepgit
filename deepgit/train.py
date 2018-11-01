# -*- coding: utf-8 -*-
"""
When this script is executed, it should train a model and write the weights
to `weights/*.pt`.
"""
from model.model import *
import torch.optim as optim
from random import choice, sample
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nb_epochs', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--embedding_dim', type=int, default=300)
args = parser.parse_args()

wi_file = open('word_index.p', 'rb')
word_to_i = pickle.load(wi_file)

training_file = open('training_pairs.p', 'rb')
repo_to_tensors = pickle.load(training_file)

args.num_embeddings = len(word_to_i)
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Y_target = torch.LongTensor([0]).to(args.device)
encoder = Encoder(args.num_embeddings,args.embedding_dim,args.hidden_dim).to(args.device)
optimizer = optim.Adam(encoder.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_samples = 5
num_repos = 100

def run(epoch, mode = "train"):
	
	total_loss = 0
	score = 0

	repos = list(repo_to_tensors.keys())
	# repos = sample(list(repo_to_tensors.keys()), num_repos)
	for r in repos:
		if len(repo_to_tensors[r]) > 1:
			base_pair = sample(repo_to_tensors[r], 2)

			base = encoder(base_pair[0].to(args.device), toggle=True)
			c1 = encoder(base_pair[1].to(args.device), toggle=False)
			similarity_scores = torch.zeros(1,num_samples+1)
			similarity_scores[0][0] = torch.dot(base[0],c1[0])

			comp_repos = repos
			comp_repos.remove(r)

			X = [choice(repo_to_tensors[choice(comp_repos)])[0][:1000] for i in range(num_samples)]
			max_len = max(x.size()[0] for x in X)
			for i, x in enumerate(X):
				X[i] = F.pad(x, (max_len-x.size()[0],0), "constant", 0)
			X = torch.stack(X,dim=0).to(args.device)

			candidate = encoder(X, toggle=False)
			for i in range(num_samples):
				# similarity_scores[0][i] = F.cosine_similarity(base,candidate[i].unsqueeze(0))
				similarity_scores[0][i] = torch.dot(base[0],candidate[i])

			Y_pred = torch.tensor(similarity_scores)

			loss = F.cross_entropy(Y_pred, torch.LongTensor([0]))
			total_loss += loss.item()
			score += int(torch.argmax(Y_pred).item() == 0)

			if mode == "train":
				loss.backward()
				torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.25)
				optimizer.step()
				optimizer.zero_grad()

	total_loss /= len(repos)
	score /= len(repos)

	print('iter', epoch, loss.item(), score)

	if mode == "test":
		torch.save(encoder, "weights/weights.epoch-%s.loss-%.03f.pt" % (epoch, loss.item()))

	torch.cuda.empty_cache()
	return loss.item(), score

for epoch in range(20):
	train_loss,train_acc = run(epoch, mode="train")
	# test_loss,test_acc = run(epoch, mode="test")
	# print(train_loss,train_acc,test_loss,test_acc)
