#!/bin/sh
for line in `cat tpid`
do
 echo $line
	tpid=`ps -ef|grep $line|grep -v grep|grep -v kill|awk '{print $2}'`
	if [ ${tpid} ]; then
	    echo 'Stop Process...'
	    kill -15 $tpid
	fi
	sleep 3
	tpid=`ps -ef|grep $line|grep -v grep|grep -v kill|awk '{print $2}'`
	if [ ${tpid} ]; then
	    echo 'Kill Process!'
	    kill -9 $tpid
	else
	    echo 'Stop Success!'
	fi
done
