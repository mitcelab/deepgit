import requests
import os

USERNAME=os.environ['gitusername']
PASSWORD=os.environ['gitpassword']

data = []
for p in range(10):
	r = requests.get("https://api.github.com/search/repositories?q=crypto size:<50000&sort=stars&order=desc&page="+str(p)+"&per_page=100", auth=(USERNAME, PASSWORD))
	# print (r.json()['items'][0].keys())
	data.extend(r.json()['items'])
	
# print (data[0]['full_name'], data[0]['name'])
names = open("github_names.txt","w")
with open("github_urls.txt","w") as urls:
	for d in data:
		urls.write(d['clone_url']+'\n')
		names.write(d['full_name'] + ' ' + str(d['watchers_count']) + ' ' + str(d['stargazers_count']) + '\n')
