# -*- coding: utf-8 -*-
"""
When this script is executed, it should train a model and write the weights
to `weights/*.pt`.
"""
from model.model import *
import torch.optim as optim
import random
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nb_epochs', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=256)
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
optimizer = optim.Adam(encoder.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

base_repo = random.choice(list(repo_to_tensors.keys()))
while len(repo_to_tensors[base_repo]) < 2: 
	base_repo = random.choice(list(repo_to_tensors.keys()))
base = encoder(repo_to_tensors[base_repo][0])
c1 = encoder(repo_to_tensors[base_repo][1])

num_samples = 5

def run(epoch, mode = "train"):
	loss = 0
	score = 0
	for r in repo_to_tensors.keys():
		if len(repo_to_tensors[r]) > 1:
			base_pair = random.sample(repo_to_tensors[r], 2)

			base = encoder(base_pair[0])
			c1 = encoder(base_pair[1])
			similarity_scores = torch.zeros(1,num_samples+1)
			similarity_scores[0][0] = F.cosine_similarity(base,c1)

			comp_repos = list(repo_to_tensors.keys())
			comp_repos.remove(r)

			for i in range(num_samples):
				candidate = encoder(random.choice(repo_to_tensors[random.choice(comp_repos)]))
				similarity_scores[0][i] = F.cosine_similarity(base,candidate)

			Y_pred = torch.tensor(similarity_scores)
			loss += F.cross_entropy(Y_pred, torch.LongTensor([0]))
			score += int(torch.argmax(Y_pred).item() == 0)
	loss /= len(repo_to_tensors)
	score /= len(repo_to_tensors)

	print('iter', epoch, loss.item(), score)
	
	if mode == "train":
		loss.backward()
		torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.25)
		optimizer.step()
		optimizer.zero_grad()

	if mode == "test":
		torch.save(encoder, "weights/weights.epoch-%s.loss-%.03f.pt" % (epoch, loss.item()))
		
	return loss.item(), score

for epoch in range(20):
	train_loss,train_acc = run(epoch, mode="train")
	test_loss,test_acc = run(epoch, mode="test")
	# print(train_loss,train_acc,test_loss,test_acc)
