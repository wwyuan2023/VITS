#!/bin/bash


# stop tts server
while true; do
    pid=$(ps -ef | grep python | grep "socket_server" | awk '{print $2}')
    if [ "x$pid" == "x" ]; then
        echo "tts server is stopped."
        break
    else
        echo "tts server is stopping..."
    fi
    while true; do
        cpid=$(ps -ef | grep python | grep $pid | awk '{print $2}')
        if [ "x$cpid" == "x" ]; then
            break
        fi
        echo "kill children process $cpid ..."
        kill $cpid >/dev/null 2>&1
        sleep 2
    done
    kill $pid >/dev/null 2>&1
    sleep 2
done

# stop http server
while true; do
    pid=$(ps -ef | grep python | grep "http_server:app" | awk '{print $2}')
    if [ "x$pid" == "x" ]; then
        echo "http server is stopped."
        break
    else
        echo "http server is stopping..."
    fi
    while true; do
        cpid=$(ps -ef | grep python | grep $pid | awk '{print $2}')
        if [ "x$cpid" == "x" ]; then
            break
        fi
        echo "kill children process $cpid ..."
        kill $cpid >/dev/null 2>&1
        sleep 2
    done
    kill $pid >/dev/null 2>&1
    sleep 2
done

# stop tlog.sh
while true; do
    pid=$(ps -ef | grep "tlog.sh" | grep -v grep | awk '{print $2}')
    if [ "x$pid" == "x" ]; then
        echo "tlog.sh is stopped."
        break
    else
        echo "tlog.sh is stopping..."
    fi
    kill $pid >/dev/null 2>&1
    sleep 1
done