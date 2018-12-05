import requests
import os
import pickle

USERNAME=os.environ['gitusername']
PASSWORD=os.environ['gitpassword']

data = []
for p in range(1,15):
	r = requests.get("https://api.github.com/search/repositories?q=topic:Cryptocurrency+size:<100000&sort=stars&order=desc&page="+str(p)+"&per_page=100", auth=(USERNAME, PASSWORD))
	if 'items' not in r.json():
		break
	data.extend(r.json()['items'])

repo_stats_d = {}
with open("raw/github_urls.txt","w") as urls:
	for d in data:
		if d['language'] and d['language'].lower() in {'python','go','c','c++','c#','javascript'}:
			urls.write(f"{d['name']},{d['clone_url']}\n")
			repo_stats_d[d['name']] = {
				'stars':d['stargazers_count'],
				'forks':d['forks_count'],
				'score':d['score'],
				'issues':d['open_issues']
			}
print (len(repo_stats_d))

dfile = open("raw/repo_stats_d.p","wb")
pickle.dump(repo_stats_d, dfile, protocol=pickle.HIGHEST_PROTOCOL)
