#!/bin/bash

if [ -z $1 ]; then
    echo "Count images in BOPdataset split."
    echo "Usage: $0 <path>"
    exit 1
fi

path=$1

for dir in "$path"/*[0-9]/rgb; do 
    count=$(ls $dir/*png | wc -l)
    fmt_count=$(printf "%4d" $count)
    echo "$(basename $(dirname $dir)):${fmt_count} images"
done

total=$(ls $path/*[0-9]/rgb/*png | wc -l)
fmt_total=$(printf "%4d" $total)
echo "Total $fmt_total"
