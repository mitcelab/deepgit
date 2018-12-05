#!/bin/bash
while read p; do
	cd ./repos
	a="$(cut -d'/' -f4 <<<"$p")"
	r="$(cut -d'/' -f5 <<<"$p")"
	fname="$a"_"$(cut -d'.' -f1 <<<"$r")"
	mkdir "$fname"
	cd "$fname"

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
