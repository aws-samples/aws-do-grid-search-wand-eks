#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Source: `pytorch imagenet example <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_ # noqa B950

Modified and simplified to make the original pytorch example compatible with
torchelastic.distributed.launch.

Changes:

1. Removed ``rank``, ``gpu``, ``multiprocessing-distributed``, ``dist_url`` options.
   These are obsolete parameters when using ``torchelastic.distributed.launch``.

2. Removed ``seed``, ``evaluate``, ``pretrained`` options for simplicity.

3. Removed ``resume``, ``start-epoch`` options.
   Loads the most recent checkpoint by default.

4. ``batch-size`` is now per GPU (worker) batch size rather than for all GPUs.

5. Defaults ``workers`` (num data loader workers) to ``0``.

Usage

::

 >>> python -m torchelastic.distributed.launch
        --nnodes=$NUM_NODES
        --nproc_per_node=$WORKERS_PER_NODE
        --rdzv_id=$JOB_ID
        --rdzv_backend=etcd
        --rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
        main.py
        --arch resnet18
        --epochs 20
        --batch-size 32
        <DATA_DIR>
"""

import argparse, io, os, shutil, time, logging, operator

from contextlib import contextmanager
from datetime import timedelta
from typing import List, Tuple
from pathlib import Path

from tqdm import tqdm
import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LinearLR

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_API_KEY'] = ''
wandb.login()

from datasets import load_dataset, Features, ClassLabel, Value, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LEDForConditionalGeneration
)

from torch.distributed.elastic.utils.data import ElasticDistributedSampler

from pathlib import Path
import pandas as pd

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from collections import OrderedDict
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

logging.getLogger().setLevel(logging.INFO)

# TODO: Refactor load/save with Huggingface from_pretrained/save_pretrained api.
# Curently SGD and ADAMW produce problems with params (like momentum)

def run(args):

    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == 0:
        wandb.init(config=args, project=args.wandb_project)
        args = wandb.config
        do_log = True
    else:
        
        do_log = False
    
    device_id = local_rank

    torch.cuda.set_device(device_id)
    logging.info(f"Set cuda device = {device_id}")

    dist.init_process_group(backend=args.dist_backend, init_method="env://", timeout=timedelta(seconds=120))

    model, criterion, optimizer = initialize_huggingface_model(
        args.arch, args.lr, args.momentum, args.weight_decay, args.optimizer, device_id
    )

    train_loader, val_loader = initialize_custom_data_loader(args.data, args.batch_size, args.workers)

    # resume from checkpoint if one exists;
    state = load_checkpoint(args.checkpoint_file, device_id, args.arch, model, optimizer)

    model_saver = SaveBestModel(args.checkpoint_file, do_log=do_log)

    # for most transformer based models Linear LR decays is best (1/10 total decay)
    scheduler = LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=args.epochs)

    # start_epoch = state.epoch + 1

    print_freq = args.print_freq

    for epoch in range(args.epochs):
        state.epoch = epoch
        train_loader.batch_sampler.sampler.set_epoch(epoch)

        logging.info(f"training epoch {epoch}")
        train_loss_epoch = train(train_loader, model, criterion, optimizer, epoch, device_id, print_freq, do_log)
        scheduler.step()

        logging.info(f"validating epoch {epoch}")
        val_loss_epoch = validate(val_loader, model, criterion, device_id, print_freq, do_log)

        if device_id == 0:
            model_saver.save(state, val_loss_epoch)

        if device_id == 0:
            save_checkpoint(state, args.checkpoint_file)

    logging.info(f"Running predictions")
    if device_id == 0:
        # del model
        # torch.cuda.empty_cache()
        run_predictions(args.checkpoint_file, args.lr, args.optimizer, do_log)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Elastic HuggingFace Training")

    # Required paramaters
    parser.add_argument(
        "--data",
        metavar="DIR",
        default="/shared-efs/wandb-finbert",
        help="path to dataset",
    )
    parser.add_argument(
        "--wandb_project",
        default="aws_eks_demo",
        help="The wandb project name",
    )
    parser.add_argument(
        "--sweep_id",
        default=None,
        help="The Sweep id created by wandb",
    )

    # Other params
    parser.add_argument("--arch", default="HuggingFace")
    parser.add_argument("--workers", default=0, type=int, help="number of data loading workers")
    parser.add_argument("--epochs", default=1, type=int, help="number of total epochs to run")
    parser.add_argument("--batch-size", default=32, type=int, help="mini-batch size per worker (GPU)")
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument("--print-freq", default=1, type=int, help="print frequency (default: 10)")
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="/shared-efs/checkpoint.pth.tar",
        help="checkpoint file path, to load and save to",
    )
    
    
    
    parser.add_argument("--optimizer", default="AdamW", help="optimizer type")

    args = parser.parse_args()

    wandb.require("service")
    wandb.setup()

    if args.sweep_id is not None:
        wandb.agent(args.sweep_id, lambda: run(args), project=args.wandb_project, count = 1)
    else:
        run(args=args)


class Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, file_name):

        #'Initialization'
        self.data_dir = data_dir
        self.df = pd.read_csv(Path(data_dir) / file_name)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        df = self.df
        one_line = df["Text"][index]
        label = df["labels"][index]

        return (one_line, label)


def collate_tokenize(data, tokenizer):
    text_batch = [element[0] for element in data]
    labels = [element[1] for element in data]
    tokenized_inputs = tokenizer(text_batch, padding="max_length", truncation=True, return_tensors="pt")

    tokenized_inputs["labels"] = torch.tensor(labels)
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"]

    return tokenized_inputs


class MyCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # do something with batch and self.params
        tokenized_inputs = collate_tokenize(batch, self.tokenizer)

        return tokenized_inputs


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 10
        self.arch = arch
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def initialize_huggingface_model(
    arch: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    optimizer_type,
    device_id: int,
):
    logging.info(f"=> creating model: {arch}")

    ## Initializing the model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.

    model.cuda(device_id)

    cudnn.benchmark = True

    model = DistributedDataParallel(model, device_ids=[device_id])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device_id)

    # initialize optimizer
    if optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return model, criterion, optimizer


def initialize_custom_data_loader(data_dir, batch_size, num_data_workers) -> Tuple[DataLoader, DataLoader]:

    # Generators
    train_dataset = Dataset(data_dir, file_name="train.csv")
    logging.info("Train dataset done")

    train_sampler = ElasticDistributedSampler(train_dataset)
    logging.info("Train sampler done")

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    my_collator = MyCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        collate_fn=my_collator,
        sampler=train_sampler,
    )

    logging.info("Train loader done")

    test_dataset = Dataset(data_dir, file_name="test.csv")

    logging.info("Test dataset done")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        collate_fn=my_collator,
    )

    logging.info("Test loader done")

    return train_loader, test_loader


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    arch: str,
    model: DistributedDataParallel,
    optimizer,  # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.
    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(arch, model, optimizer)
    
    print('***** Checkpoint File = '+checkpoint_file)

    if os.path.isfile(checkpoint_file):
        logging.info(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        logging.info(f"=> loaded checkpoint file: {checkpoint_file}")

    logging.info(f"=> done restoring from previous checkpoint")
    return state


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


class SaveBestModel:
    "A simple model saver Callback"

    def __init__(self, filename, min_metric=True, do_log=True):
        self.filename = filename
        self.min_metric = min_metric
        self.do_log = do_log
        self.checkpoint_dir = os.path.dirname(filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best = 100 if min_metric else -1

    def save(self, state, metric_value):
        torch.save(state.capture_snapshot(), self.filename)
        op = operator.lt if self.min_metric else operator.gt
        if op(metric_value, self.best):
            logging.info(f"=> best model found at epoch {state.epoch}")
            self._save()

    def _save(self):
        best_model = os.path.join(self.checkpoint_dir, "model_best.pth.tar")
        shutil.copyfile(self.filename, best_model)
        if self.do_log:
             self.log_model(best_model)

    def log_model(self, path, metadata={}, description="trained model"):
        "Log model file"
        if wandb.run is None:
            raise ValueError("You must call wandb.init() before log_model()")
        path = Path(path)
        if not path.is_file():
            raise f"path must be a valid file: {path}"
        name = f"run-{wandb.run.id}-model"
        artifact_model = wandb.Artifact(name=name, type="model", metadata=metadata, description=description)
        with artifact_model.new_file(name, mode="wb") as fa:
            fa.write(path.read_bytes())
        wandb.run.log_artifact(artifact_model)


def save_checkpoint(state: State, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    #tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), filename)
    #os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    # if is_best:
    #     best = os.path.join(checkpoint_dir, "model_best.pth.tar")
    #     print(f"=> best model found at epoch {state.epoch} saving to {best}")
    #     shutil.copyfile(filename, best)


def train(
    train_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    optimizer,  # AdamW,
    epoch: int,
    device_id: int,
    print_freq: int,
    do_log: bool,
):
    losses = AverageMeter("Loss", ":.4e")

    model.train()

    for batch in tqdm(train_loader, total=len(train_loader)):

        optimizer.zero_grad()
        input_ids = batch["input_ids"].cuda(device_id, non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(device_id, non_blocking=True)
        labels = batch["labels"].cuda(device_id, non_blocking=True)

        # forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # hf models return loss as 1st argument
        loss = outputs.loss
        if do_log:
            wandb.log({"train_loss": loss.item()})

        # # measure accuracy and record loss
        losses.update(loss.item(), input_ids.size(0))

        loss.backward()
        optimizer.step()

    return losses.avg


def validate(
    val_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    device_id: int,
    print_freq: int,
    do_log: bool,
):
    losses = AverageMeter("Loss", ":.4e")

    metrics = [Metric("val_recall", recall_score), 
               Metric("val_f1",f1_score),
               Metric("val_accuracy", accuracy_score),
               Metric("val_precision", precision_score)]

    # switch to evaluate mode
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(val_loader, total=len(val_loader)):

            input_ids = batch["input_ids"].cuda(device_id, non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(device_id, non_blocking=True)
            labels = batch["labels"].cuda(device_id, non_blocking=True)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # compute output
            loss = outputs[0]

            # # measure accuracy and record loss
            losses.update(loss.item(), input_ids.size(0))

            #compute metrics
            pred_labels = outputs.logits.argmax(axis=1).cpu()
            true_labels = labels.cpu()
            
            for m in metrics:
                m.update(pred_labels, true_labels)

        if do_log:
            wandb.log({"val_loss": losses.avg})
            wandb.log({m.name:m.avg for m in metrics})

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class Metric(AverageMeter):

    def __init__(self, name, func):
        super().__init__(name)
        self.func = func
    
    def update(self, y_pred, y_true):
        val = self.func(y_pred=y_pred, y_true=y_true)
        super().update(val)


# def accuracy(output, target, topk=(1,)):
#     """
#     Computes the accuracy over the k top predictions for the specified values of k
#     """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


def run_predictions(checkpoint_filename, lr, optimizer, do_log):
    print("**********************\nRunning predictions\n**********************")
    
    checkpoint_dir = os.path.dirname(checkpoint_filename)
    logging.info(checkpoint_dir)
    
    # Run inference on CPU to avoid Cuda out of memory issue
    # device = torch.device("cuda")
    device = torch.device("cpu")

    model_name = "bert-base-cased"
    test_file = "/shared-efs/wandb-finbert/test.csv"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    checkpoint = torch.load(checkpoint_dir + "/checkpoint.tar")
    state_dict = checkpoint["state_dict"]
    # create new OrderedDict that does not contain `module.`

    logging.info("Doing some state dict magic")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = k.replace("module.", "")  # removing ‘moldule.’ from key
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_df = pd.read_csv(test_file)
    
    print('****Len test_df = '+str(test_df.shape[0]))

    logging.info("Tokening inputs")
    tokenized_test_inputs = tokenizer(
        list(test_df["Text"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    
    tokenized_test_inputs.to(device)
    model.to(device)
    logging.info(f"Running inference on: {device}")

    model.eval()
    with torch.no_grad():
        # preds = model(**tokenized_test_inputs)
        preds = model(tokenized_test_inputs['input_ids'], tokenized_test_inputs['attention_mask'])

    pred_labels = preds.logits.argmax(axis=1).tolist()
    true_labels = list(test_df["labels"])

    recall = recall_score(y_pred=pred_labels, y_true=true_labels)
    f1 = f1_score(y_pred=pred_labels, y_true=true_labels)
    accuracy = accuracy_score(y_pred=pred_labels, y_true=true_labels)
    precision = precision_score(y_pred=pred_labels, y_true=true_labels)

    pred_dict = {
        "test_recall": recall,
        "test_f1": f1,
        "test_accuracy": accuracy,
        "test_precision": precision,
    }

    metrics_df = pd.DataFrame()
    metrics_df = metrics_df.append(pred_dict, ignore_index=True)

    run_name = checkpoint_dir.split("/")[-1]
    metrics_df["run_name"] = run_name
    metrics_df["lr"] = lr
    metrics_df["optimizer"] = optimizer
    
    print(metrics_df)

    out_file = "all_results.csv"

    logging.info(f"Logging metrics to : {out_file}")
    if os.path.exists(f"/shared-efs/wandb-finbert/{out_file}"):
        metrics_df.to_csv(f"/shared-efs/wandb-finbert/{out_file}", mode="a", index=False, header=False)
    else:
        metrics_df.to_csv(f"/shared-efs/wandb-finbert/{out_file}", index=False)
        
    if do_log:
        wandb.log(pred_dict)

    return None


if __name__ == "__main__":
    main()
