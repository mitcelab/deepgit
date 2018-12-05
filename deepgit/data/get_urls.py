import requests
import os
import pickle

USERNAME=os.environ['gitusername']
PASSWORD=os.environ['gitpassword']

data = []
for p in range(1,4):
	r = requests.get("https://api.github.com/search/repositories?q=topic:Cryptocurrency+language:python&language:go&language:cpp&language:c+stars:>100+size:<100000&sort=stars&order=desc&page="+str(p)+"&per_page=100", auth=(USERNAME, PASSWORD))
	data.extend(r.json()['items'])

count_d = {}
with open("github_urls.txt","w") as urls:
	for d in data:
		urls.write(d['clone_url']+'\n')
		count_d[d['name']] = [d['watchers_count'], d['stargazers_count']]

dfile = open("github_count_d.p","wb")
pickle.dump(count_d, dfile, protocol=pickle.HIGHEST_PROTOCOL)
