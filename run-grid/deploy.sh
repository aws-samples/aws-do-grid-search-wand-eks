#!/bin/bash

kubectl apply -k ./train/config/default

kubectl -n elastic-job get pods

kubectl apply -f ./train/config/samples/etcd.yaml

