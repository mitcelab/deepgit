import requests
from credentials import *

r = requests.get("https://api.github.com/search/repositories?q=any&sort=stars&order=desc", auth=(USERNAME, PASSWORD))
data = r.json()

# print len(data['items'])
# print [k for k in data['items'][0]]
with open("github_urls.txt","wb") as urls:
	for d in data['items']:
		urls.write(d['clone_url']+'\n')
