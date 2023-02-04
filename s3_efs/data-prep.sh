#!/bin/bash

cd ./shared-efs
mkdir wandb-finbert
cd ./wandb-finbert

echo "copying ......."

aws s3 cp s3://wandb-finbert/stock_data.csv ./

echo "done ......"

