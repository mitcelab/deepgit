#!/bin/bash
while read p; do
	cd ./repos
	f="$(cut -d'/' -f5 <<<"$p")"
	echo "$f"
	mkdir "$f"
	cd "$f"
	
	for i in `seq 1 4`;
	do
		pwd
		mkdir "$i"
		cd "$i"
		pwd
		git clone "$p" .
		echo clone
		git checkout HEAD~"$i"
		echo checkout
		cd ..
	done
	cd ..
done <github_urls.txt
