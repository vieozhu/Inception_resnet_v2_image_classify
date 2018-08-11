#!/bin/sh
myPath="./log"
if [ ! -d "$myPath" ]; then
  mkdir $myPath
fi

rm -f tpid
nohup ~/software/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9001 --model_config_file=/data3/diagnose-model/model_config_file.json  > $myPath/startup`date +%Y-%m-%d`.log 2>&1 &
echo $! > tpid
echo Start Success!
#nohup ~/software/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9001 --model_config_file=/data3/diagnose-model/model_config_file.json > nohup.log &
