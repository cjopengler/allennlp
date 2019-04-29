#!/usr/bin/env bash

serialize=~/serialize/person_name_classify

train_json=projects/training_config/person_name_classify.jsonnet



echo ${serialize}
echo ${train_json}
rm -fr ${serialize}
mkdir -p ${serialize}


allennlp \
train ${train_json} \
-s ${serialize} \
--include-package projects