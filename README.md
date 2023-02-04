## Accelerate Deep Learning Hyperparameter Grid Search with Amazon EKS and Weights & Biases

# 1. Overview
Hyperparameter optimization is highly computationally demanding for deep learning models. The architectural complexity increases when a single model training run requires multiple GPUs. In this repo, we will show how to use the Weights & Biases (W&B) Sweeps function and Amazon’s Elastic Kubernetes Service (EKS) to address these challenges. Amazon EKS is a highly available managed Kubernetes service that automatically scales instances based on load, and is well suited for running distributed training workloads. We will showcase an example for tuning a bert-base-cased model for classifying positive or negative sentiment for stock market data headlines. In the following sections, we will present the key components of the architecture shown in Fig. 1. More specifically, we will show:

1. How to set up an EKS cluster with a scalable file system
2. How to train PyTorch models using TorchElastic
3. Present a solution architecture integrating W&B with EKS and TorchElastic

<div align="center">
<img src="./Achitecture.png" width="90%">
<br/>
Fig. 1 - Sample EKS infrastructure for hyperparameter grid search with deep learning models
</div>
<br/>



TODO: Fill this README out!

Be sure to:

* Change the title in this README
* Edit your repository description on GitHub

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information. Prior to any production deployment, customers should work with their local security teams to evaluate any additional controls



## License

This library is licensed under the MIT-0 License. See the LICENSE file.

