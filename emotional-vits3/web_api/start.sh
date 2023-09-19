#!/bin/bash

num_gpus=1
num_jobs=10

if [ -n "$1" ]; then
  num_gpus=$1
fi
if [ -n "$2" ]; then
  num_jobs=$2
fi

# check tts server
pid=$(ps -ef | grep python | grep "socket_server" | awk '{print $2}')
if [ "x$pid" == "x" ]; then
    echo "tts server is starting."
    nohup python3 socket_server.py --n-gpus $num_gpus -j $num_jobs --loglv 1 >tts.log 2>&1 &
    sleep 10
else
    echo "tts server is already running."
fi

# check http server
pid=$(ps -ef | grep python | grep "http_server:app" | awk '{print $2}')
if [ "x$pid" == "x" ]; then
    echo "http server is starting."
    nohup uvicorn http_server:app --host 0.0.0.0 --port 10009 --workers `expr $num_jobs \* $num_gpus` >http.log 2>&1 &
    sleep 1
else
    echo "http server is already running."
fi

# check tlog
pid=$(ps -ef | grep "tlog.sh" | grep -v grep | awk '{print $2}')
if [ "x$pid" == "x" ]; then
    echo "tlog is starting."
    nohup bash tlog.sh >/dev/null 2>&1 &
    sleep 1
else
    echo "tlog is already running."
fi

