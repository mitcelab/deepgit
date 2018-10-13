class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.randn(1, self.hidden_size)

# make a tensor for each file in a list of given files
def makeTensors(files):
	word_counts = {}
	word_lists = {}

	# get word_counts across all files, word_list of each file
	for name,_ in files:
		f = open(name)
		for line in f.readlines():
			for word in re.findall(r"[\w']+|[.,!?;:]", line.lower()):
				if word in word_counts:
					word_counts[word] += 1
				else:
					word_counts[word] = 1

				if name in word_lists:
					word_lists[name].append(word)
				else:
					word_lists[name] = [word]


	# create word to index of words across all files
	for word, count in word_counts.items():#Counter(word_counts).most_common(64000):
		word_to_i[word] = len(word_to_i)

	# create tensors 
	# list[ (doc_tensor,document) ]
	dim = len(word_lists[max(word_lists, key=lambda x: len(word_lists[x]))])
	training_pairs = []
	for name,repo in files:
		doc_tensor = torch.zeros(dim, 1, len(word_to_i))
		for wi, word in enumerate(word_lists[name]):
			doc_tensor[wi][0][word_to_i[word]] = 1
		output_tensor = torch.tensor([repo], dtype=torch.long)
		training_pairs.append((doc_tensor, output_tensor))
	return training_pairs


criterion = nn.NLLLoss()
learning_rate = 0.005

def train(target, input):
	hidden = rnn.initHidden()
	rnn.zero_grad()
	for i in range(input.size()[0]):
		output,hidden = rnn(input[i],hidden)

	print(output,target)

	loss = criterion(output, target)
	loss.backward()

	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.item()


def get_files():
	file_repo = []
	for root,dirs,files in os.walk("./websocketd"):
		for file in files:
			if file.endswith('.go'):
				file_repo.append((root + '/' + file,0))
	for root,dirs,files in os.walk("./salt"):
		for file in files:
			if file.endswith('.py') and len(file_repo) < 30:
				file_repo.append((root + '/' + file,1))
	return file_repo



files = get_files()
# training_pairs = makeTensors(files)
# training_pairs = makeTensors([('crypt.py',0), ('http.go',1), ('fileclient.py',0), ('console.go',1)])
# rnn = RNN(len(word_to_i), 256, 2)
# for i in range(10):
# 	for pair in training_pairs:
# 		output,loss = train(pair[1],pair[0])
# 		print (loss)
# print(output)