#!/bin/bash

wdir=`pwd`;
while true; do
    # truncate log file
    find $wdir -name "*.log" -type f -exec truncate -s 10M {} \;
    # sleep 1min
    sleep 60;
done
