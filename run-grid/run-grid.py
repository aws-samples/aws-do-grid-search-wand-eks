from json import load
import os, logging, yaml

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

import wandb

os.environ['WANDB_API_KEY'] = 'enter-wandb-api-key'
wandb.login()

ryaml = YAML()
ryaml.preserve_quotes = True

logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#GLOBALS
TRAIN_CONFIG = './train.yaml'
SWEEP_CONFIG = './sweep_config.yaml'
WANDB_PROJECT_NAME = "aws_eks_demo"
SWEEP_ID = ""


with open(SWEEP_CONFIG) as file:
    sweep_config = yaml.full_load(file)

with open(TRAIN_CONFIG) as file:
    train_config = ryaml.load(file)

#create sweep controller
SWEEP_ID = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
logging.info(f"Creating Sweep: {SWEEP_ID}")

# let's write the sweep_id on the file as arg for the main script
def update_sweep_info(sweep_id, project):
    "Inject wandb info in train yaml"
    train_config["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][6] = dq(f'--wandb_project={project}')
    train_config["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][7] = dq(f'--sweep_id={sweep_id}')
    with open(TRAIN_CONFIG, 'w') as file:
        ryaml.dump(train_config, file)

update_sweep_info(SWEEP_ID, WANDB_PROJECT_NAME)

if not os.path.isdir("./train-yamls"):
    os.mkdir("./train-yamls")
    


nruns = 6

for i in range(nruns):
    
    checkpoint_file = '/shared-efs/wandb-finbert/job-'+SWEEP_ID+'/run-'+str(i)+'/checkpoint.tar'
    
    train_config["metadata"]["name"] = 'wandb-finbert-job-'+str(i)
    train_config["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][8] = dq(f'--checkpoint-file={checkpoint_file}')
    
    job_yaml_file = './train-yamls/train-'+str(i)+'.yaml'
    with open(job_yaml_file, 'w') as file:
        ryaml.dump(train_config, file)
    

    
    

