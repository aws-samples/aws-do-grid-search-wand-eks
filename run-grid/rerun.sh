#!/bin/bash

cd ./train-yamls/
for FILE in *;
    do echo $FILE;
    kubectl delete -f $FILE
done

rm *.yaml
cd ..


kubectl delete -f ./train/config/samples/etcd.yaml

sleep 2

kubectl apply -f ./train/config/samples/etcd.yaml

sleep 2

cd ./train
./build.sh

sleep 2

./push.sh

sleep 2

