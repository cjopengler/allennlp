#!/usr/bin/env bash

serialize=~/serialize/academic_paper_classify
serialize=~/serialize/academic_paper_classify2

train_json=projects/training_config/academic_paper_classifier.json
train_json=projects/training_config/academic_paper_classifier2.json



echo ${serialize}
echo ${train_json}
rm -fr ${serialize}
mkdir -p ${serialize}


allennlp \
train ${train_json} \
-s ${serialize} \
--include-package projects