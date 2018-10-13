import requests
import os

USERNAME=os.environ['gitusername']
PASSWORD=os.environ['gitpassword']

data = []
for p in range(10):
	r = requests.get("https://api.github.com/search/repositories?q=crypto&size<10000&sort=stars&order=desc&page="+str(p)+"&per_page=100", auth=(USERNAME, PASSWORD))
	print (len(r.json()['items']))
	data.extend(r.json()['items'])
	
print (len(data))
with open("github_urls.txt","w") as urls:
	for d in data:
		urls.write(d['clone_url']+'\n')
