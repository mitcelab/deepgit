#!/bin/bash
while read p; do
	cd ./repos
	fname="$(cut -d',' -f1 <<<"$p")"
	url="$(cut -d',' -f2 <<<"$p")"
	mkdir "$fname"
	cd "$fname"

	for i in `seq 1 4`;
	do
		pwd
		mkdir "$i"
		cd "$i"
		pwd
		git clone "$url" .
		echo clone
		git checkout HEAD~"$i"
		echo checkout
		cd ..
	done
	cd ..
done <github_urls.txt
