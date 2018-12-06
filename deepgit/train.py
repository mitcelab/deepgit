# -*- coding: utf-8 -*-
"""
When this script is executed, it should train a model and write the weights
to `weights/*.pt`.
"""
import gc
from model.model import *
import torch.optim as optim
from random import choice, sample
import pickle
import argparse
from statistics import mean


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--candidates', type=int, default=99)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument("--checkpoint", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1)
args = parser.parse_args()

wi_file = open('data/processed/word_index.p', 'rb')
word_to_i = pickle.load(wi_file)

training_file = open('data/processed/training_pairs.p', 'rb')
repo_to_tensors = pickle.load(training_file)

args.num_embeddings = len(word_to_i)
args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(args.device)

Y_target = torch.LongTensor([0]).to(args.device)
encoder = Encoder(args.num_embeddings,args.embedding_dim,args.hidden_dim).to(args.device) if args.checkpoint<1 else torch.load(f'weights/weights.epoch-{args.checkpoint}.pt')
optimizer = optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_samples = args.candidates
print (len(repo_to_tensors), num_samples)

def run(epoch, mode = "train"):
	total_loss = 0
	score,score5 = [],[]

	repos = list(repo_to_tensors.keys())
	for r in repos:
		try:
			if len(repo_to_tensors[r]) > 1:
				# calculate base repository
				base_pair = sample(repo_to_tensors[r], 2)
				base = encoder(base_pair[0].to(args.device), toggle=True)
				c1 = encoder(base_pair[1].to(args.device), toggle=False)
				similarity_scores = torch.zeros(1,num_samples+1)
				similarity_scores[0][0] = torch.dot(base[0],c1[0])

				# get other repositories
				comp_repos = repos
				comp_repos.remove(r)

				X = [choice(repo_to_tensors[choice(comp_repos)])[0][:1000] for i in range(num_samples)]
				max_len = max(x.size()[0] for x in X)
				for i, x in enumerate(X):
					X[i] = F.pad(x, (max_len-x.size()[0],0), "constant", 0)
				X = torch.stack(X,dim=0).to(args.device)

				candidate = encoder(X, toggle=False)

				# calculate similarities for all candidates
				for i in range(num_samples):
					# similarity_scores[0][i] = F.cosine_similarity(base,candidate[i].unsqueeze(0))
					similarity_scores[0][i+1] = torch.dot(base[0],candidate[i])
				Y_pred = torch.tensor(similarity_scores)

				loss = F.cross_entropy(Y_pred, torch.LongTensor([0]))
				total_loss += loss.item()
				score.append(int(torch.argmax(Y_pred).item() == 0))
				score5.append(int(0 in torch.topk(Y_pred,k=5)[1][0]))

				if mode == "train":
					loss.backward()
					torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.25)
					optimizer.step()
					optimizer.zero_grad()

		except Exception as e:
			print (e)
		finally:
			gc.collect()
			torch.cuda.empty_cache()

	total_loss /= len(repos)

	if mode == "test":
		torch.save(encoder, f'weights/weights.epoch-{epoch}.pt')
		scheduler.step(total_loss)

	print('iter:', epoch, mode, mean(score), mean(score5), loss.item(), total_loss)

	gc.collect()
	torch.cuda.empty_cache()
	return total_loss, mean(score), mean(score5)


loss_f = open('logs/train_losses.txt','w')
test_f = open('logs/test_losses.txt', 'w')
for epoch in range(args.checkpoint+1,args.checkpoint+1+args.epochs):
	train_total, train_acc1, train_acc5 = run(epoch, mode="train")
	loss_f.write(f'{epoch} : {train_total} : {train_acc1} : {train_acc5}\n')
	test_total, test_acc1, test_acc5 = run(epoch, mode="test")
	test_f.write(f'{epoch} : {test_total} : {test_acc1} : {test_acc5}\n')
