#!/bin/bash

while read p; do
	cd ~/deepgit/data
	git clone "$p"
done <github_urls.txt