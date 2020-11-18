#!/bin/bash
for filename in ../testcases/*.txt; do
    name=${filename##*/}
    echo "$name"
    python3 pageRank.py "$filename" directed pagerank_directed_"$name"
done
