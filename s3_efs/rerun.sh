#!/bin/bash

kubectl delete -f efs-data-prep-pod.yaml

./build.sh

./push.sh


