#!/bin/bash

while read p; do
	cd ./repos
	git clone "$p" 
done <github_urls.txt