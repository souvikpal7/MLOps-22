#!/bin/bash
echo Your container args are: "$@"
# echo "$1"
# echo "$2"

python hand_written_digits.py --clf_name $1 --random_state $2
