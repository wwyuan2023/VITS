#!/bin/bash

root=$1
maxno=$2
if [ "x${root}" == "x" ]; then
    echo "Root directory should be valid!";
    exit 1;
fi
if [ "x${maxno}" == "x" ]; then
    maxno=3
fi
echo "root=$root, maxno=$maxno"

while true; do
    find $root -name "*.pth" | perl -e '
        $maxno = int($ARGV[0]) * 2;
        while(<STDIN>){
            chomp;
            /[GD]_(\d+)\.pth/; 
            push @a, [$1, $_];
        } 
        @b = sort {$b->[0] <=> $a->[0]} @a; 
        for($i=$maxno; $i<=$#b; $i++){
            print "delete $b[$i]->[1]\n";
            unlink $b[$i]->[1];
        }
    ' $maxno; 
    sleep 300;
done