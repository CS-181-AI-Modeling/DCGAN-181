#!/bin/bash

folder="wikiart"

# Find duplicate file names and remove all but one
find "${folder}" -type f -printf '%f\n' | sort | uniq -d | while read -r duplicate_name; do
    duplicate_files=$(find "${folder}" -type f -name "${duplicate_name}")
    first=1
    for file in $duplicate_files; do
        if [ $first -eq 1 ]; then
            first=0
        else
            rm "${file}"
        fi
    done
done

echo "Duplicate files removed."
