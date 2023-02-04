import os
import wandb
import random
import pandas as pd

os.environ['WANDB_API_KEY'] = 'enter-wandb-api-key'

raw_csv_fname = "/shared-efs/wandb-finbert/stock_data.csv"

wandb.login()
wandb.init(project="aws_demo", job_type="data_upload")

table = wandb.Table(columns=["Sequence", "Sentiment"])

ds_at = wandb.Artifact("raw_dataset", type="dataset")
ds_at.add_file(raw_csv_fname)

wandb.log_artifact(ds_at)
wandb.finish()

#Pre-process Data
wandb.init(project="aws_demo", job_type="preprocess_data")
dataset_path = wandb.use_artifact("raw_dataset:latest").download()
raw_csv_fname = os.path.join(dataset_path, raw_csv_fname)

df = pd.read_csv(raw_csv_fname)

labels = ["negative", "positive"]
id2label = {-1: labels[0], 1: labels[1]}
label2id = { labels[0]:-1, labels[1]:1 }

df["labels"] = df["Sentiment"].map({-1:0, 1:1})

df = df.drop(columns=["Sentiment"])

def get_train_test_idxs(df, pct=0.1, seed=2022):
    "get train and valid idxs"
    random.seed(seed)
    range_of = lambda df: list(range(len(df)))
    test_idxs = random.sample(range_of(df), int(pct*len(df)))
    train_idxs = [i for i in range_of(df) if i not in test_idxs]
    return train_idxs, test_idxs
    
def save_datasets(df, pct=0.1):
    "Save splitted dataset"
    train_idxs, test_idxs = get_train_test_idxs(df, pct)
    train_df, test_df = df.loc[train_idxs], df.loc[test_idxs]
    print("Saving splitted dataset")
    train_df.to_csv("/shared-efs/wandb-finbert/train.csv", index=False)
    test_df.to_csv("/shared-efs/wandb-finbert/test.csv", index=False)
    
save_datasets(df, pct=0.1)

split_at = wandb.Artifact("splitted_dataset", type="dataset")

# we add the files
split_at.add_file("/shared-efs/wandb-finbert/train.csv")
split_at.add_file("/shared-efs/wandb-finbert/test.csv")

# we log
wandb.log_artifact(split_at)

wandb.finish()
