#!/bin/bash

src="wikiart/wikiart"

# Move all images from subdirectories to the parent directory
find "${src}"/* -type f -exec mv -t "${src}" {} +

# Remove empty subdirectories
find "${src}" -mindepth 1 -type d -empty -delete
