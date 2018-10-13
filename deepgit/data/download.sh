#!/bin/bash

while read p; do
	cd /Volumes/Aaron\'s\ Drive/deepgit
	git clone "$p" 
done <github_urls.txt