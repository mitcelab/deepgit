import re
import torch
from collections import Counter
import os
import pickle
#import matplotlib.pyplot as plt

word_to_i = {'UNKNOWN':0, 'INTEGER':1}
repo_to_file = {}
repo_dict = {}

def makeRepoFile(path):
	for d in os.listdir(path):
		if (os.path.isdir(path+d)):
			file_history = {}
			for d2 in os.listdir(path+d):
				# print (path + d + '/' +d2)
				files = getFiles(path + d + '/' + d2)
				# print (files)
				if files:
					file_history[d2] = files
			if file_history:
				print (len(file_history[d2]))
				repo_to_file[d] = file_history
	print (len(repo_to_file))

def getFiles(path):
	result = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith('.py') or file.endswith('.go') or file.endswith(".c") or file.endswith(".cpp"):
				result.append(root + '/' + file)
	return result

def makeRepoDict():
	for r in repo_to_file.keys():
		if r not in repo_dict:
			repo_dict[r] = len(repo_dict)

# make a tensor for each file in a list of given files
def makeTensors():
	word_counts = {}
	word_lists = {}

	# get word_counts across all files, word_list of each file
	for _,file_d in repo_to_file.items():
		for _,version in file_d.items():
			for path in version:
				# name = path.split('/')[-1]
				try:
					f = open(path)
					assert path not in word_lists
					word_lists[path] = []

					for line in f.readlines():
						for word in re.findall(r"[\w']+|[=().,!?;:]", line.lower()):
							if word in word_counts:
								word_counts[word] += 1
							else:
								word_counts[word] = 1
							word_lists[path].append(word)

				except Exception as e:
					print('skip', path, e)

	# create word to index of words across all files
	for word, count in Counter(word_counts).most_common(6400):
		word_to_i[word] = len(word_to_i)

	# create tensor for each file
	repo_to_tensor = {}
	count = 0
	for repo,version in repo_to_file.items():
		for _,files in version.items():
			training_pairs = []
			for path in files:
				if path in word_lists and word_lists[path]:
					doc_tensor = torch.zeros(1,len(word_lists[path]), dtype=torch.long)
					for wi, word in enumerate(word_lists[path]):
						if word in word_to_i:
							doc_tensor[0][wi] = word_to_i[word]
						elif str.isdigit(word):
							doc_tensor[0][wi] = word_to_i['INTEGER']
						else:
							word_to_i['UNKNOWN']
					training_pairs.append(doc_tensor)
			repo_to_tensor[repo] = training_pairs
	return repo_to_tensor

makeRepoFile('/home/paperspace/repos/')
# makeRepoFile('/Users/aaronhuang/repos/')
makeRepoDict()
repo_to_tensor = makeTensors()

wi_file = open('data/processed/word_index.p', 'wb')
pickle.dump(word_to_i, wi_file, protocol=pickle.HIGHEST_PROTOCOL)

training_file = open('data/processed/training_pairs.p', 'wb')
pickle.dump(repo_to_tensor, training_file, protocol=pickle.HIGHEST_PROTOCOL)
