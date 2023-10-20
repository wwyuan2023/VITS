#!/bin/bash


# check installation
if ! command -v denoise8k-infer > /dev/null; then
    echo "Error: Please setup denoise8k." >&2
    return 1
fi
if ! command -v text-parser > /dev/null; then
    echo "Error: Please setup text-parser." >&2
    return 1
fi

# prepare tools
ln -sf ../toolkits
find . -name "*.sh" -type f -exec chmod +x {} \;
find . -name "*.py" -type f -exec chmod +x {} \;
find . -name "*.pl" -type f -exec chmod +x {} \;

