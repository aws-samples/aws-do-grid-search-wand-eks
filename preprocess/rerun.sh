#!/bin/bash

kubectl delete -f wandb-preprocess.yaml

./build.sh

./push.sh

