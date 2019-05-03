#!/usr/bin/env bash

serialize=~/serialize/trigger_detection

train_json=projects/data/event/trigger_detection.jsonnet


echo ${serialize}
echo ${train_json}
rm -fr ${serialize}
mkdir -p ${serialize}


allennlp \
train ${train_json} \
-s ${serialize} \
--include-package projects