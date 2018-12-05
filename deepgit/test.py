import pickle
import torch
import argparse
import gc
from model.model import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

pairs_file = open('data/processed/training_pairs.p', 'rb')
repo_to_tensors = pickle.load(pairs_file)

stats_f = open('data/raw/repo_stats_d.p','rb')
stats_d = pickle.load(stats_f)

def make_input():
	X = []
	repo_endpoints = {}
	repos = list(repo_to_tensors.keys())
	for r in repos:
		for tensor in repo_to_tensors[r][:10]:
			start = len(X)
			X.append(tensor[0][:1000])
			end = len(X)
		repo_endpoints[r] = (start,end)

	max_len = max(x.size()[0] for x in X)
	for i,x in enumerate(X):
		X[i] = F.pad(x, (max_len-x.size()[0],0), "constant", 0)
	return (X,repo_endpoints)

def encode_inputs(X, checkpoint = 'weights/weights.epoch-0.loss-0.874.pt'):
	encoder = torch.load(checkpoint)
	encoded = None
	for b in range(8):
		if b*args.batch_size >= len(X):
			break
		batch = torch.stack(X[b*args.batch_size:(b+1)*args.batch_size], dim=0)
		curr = torch.cat((encoder(batch, toggle=False),encoder(batch, toggle=True)), dim=1)
		if encoded is None:
			encoded = curr
		else:
			encoded = torch.cat((encoded,curr), dim=0)
		torch.cuda.empty_cache()
		gc.collect()
	return encoded

def combine_repo(repo_endpoints, encoded, combine_func=lambda x:torch.mean(x,0)):
	X,Y = [],[]
	for r,e in repo_endpoints.items():
		X.append(combine_func(encoded[e[0]:e[1]]))
		Y.append(stats_d[r])
	return (X,Y)

X,repo_endpoints = make_input()
encoded = encode_inputs(X)
X,Y = combine_repo(repo_endpoints, encoded)
print(X,Y)

# labels_f = open('labels.p','wb')
# pickle.dump(Y[:800], labels_f, protocol=pickle.HIGHEST_PROTOCOL)
#
# encoded = open('vecs.p','wb')
# pickle.dump(xl,encoded, protocol=pickle.HIGHEST_PROTOCOL)
