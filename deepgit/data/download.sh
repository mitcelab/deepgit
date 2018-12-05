#!/bin/bash
while read p; do
	cd ./repos
	for i in `seq 1 10`;
	do
		pwd
		cd "$i"
		git clone "$p"
		git checkout HEAD~"$i"
		cd ..
	done
done <urls.txt
