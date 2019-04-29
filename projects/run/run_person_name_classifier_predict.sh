#!/usr/bin/env bash

allennlp predict \
~/serialize/person_name_classify/model.tar.gz \
projects/data/person_name_classify/person_name.test.data \
--output-file projects/data/person_name_classify/person_name.test.predict.data \
--include-package projects \
--predictor PersonNamePredictor