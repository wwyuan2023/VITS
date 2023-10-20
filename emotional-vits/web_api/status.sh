#!/bin/bash


# status tts server
pid=$(ps -ef | grep python | grep "socket_server" | awk '{print $2}')
if [ "x$pid" == "x" ]; then
    echo "tts server is stopped."
else
    echo "tts server is running."
fi

# status http server
pid=$(ps -ef | grep python | grep "http_server:app" | awk '{print $2}')
if [ "x$pid" == "x" ]; then
    echo "http server is stopped."
else
    echo "http server is running."
fi

# status tlog.sh
pid=$(ps -ef | grep "tlog.sh" | grep -v grep | awk '{print $2}')
if [ "x$pid" == "x" ]; then
    echo "tlog.sh is stopped."
else
    echo "tlog.sh is running."
fi
