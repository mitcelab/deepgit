import re
import torch
from collections import Counter
import os
import pickle

word_to_i = {'UNKNOWN':0}
repo_to_file = {}
repo_dict = {}

def makeRepoFile(path):
	for d in os.listdir(path):
		files = getFiles(path + d)
		if files:
			repo_to_file[d] = files

def getFiles(path):
	result = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith('.py') or file.endswith('.go'):
				result.append(root + '/' + file)
	return result
	
def makeRepoDict():
	for r,files in repo_to_file.items():
		if r not in repo_dict:
			repo_dict[r] = len(repo_dict)

# make a tensor for each file in a list of given files
def makeTensors():
	word_counts = {}
	word_lists = {}

	# get word_counts across all files, word_list of each file
	for _,files in repo_to_file.items():
		for path in files:
			# name = path.split('/')[-1]
			f = open(path)
			try:
				assert path not in word_lists
				word_lists[path] = []

				for line in f.readlines():
					for word in re.findall(r"[\w']+|[.,!?;:]", line.lower()):
						if word in word_counts:
							word_counts[word] += 1
						else:
							word_counts[word] = 1
						word_lists[path].append(word)

			except Exception as e:
				print('skip', path, e)
	print (Counter(word_counts).most_common(100))
	# create word to index of words across all files
	for word, count in Counter(word_counts).most_common(6400):
		word_to_i[word] = len(word_to_i)

	# create tensors 
	# list[ (doc_tensor,document) ]
	dim = len(word_lists[max(word_lists, key=lambda x: len(word_lists[x]))])
	dim = min(dim, 10240)
	
	repo_to_tensor = {}
	count = 0
	for repo,files in repo_to_file.items():
		training_pairs = []
		for path in files:
			doc_tensor = torch.zeros(1,dim, dtype=torch.long)
			if path in word_lists:
				for wi, word in enumerate(word_lists[path][:dim]):
					doc_tensor[0][wi] = word_to_i[word] if word in word_to_i else word_to_i['UNKNOWN']
				training_pairs.append(doc_tensor)
		repo_to_tensor[repo] = training_pairs
	return repo_to_tensor

makeRepoFile('./data/repos/')
makeRepoDict()
training_pairs = makeTensors()
print (training_pairs.keys())

wi_file = open('word_index.p', 'wb')
pickle.dump(word_to_i, wi_file, protocol=pickle.HIGHEST_PROTOCOL)

training_file = open('training_pairs.p', 'wb')
pickle.dump(training_pairs, training_file, protocol=pickle.HIGHEST_PROTOCOL)

