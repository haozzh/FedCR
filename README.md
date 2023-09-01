# FedCR: Personalized Federated Learning Based on Across-Client Common Representation with Conditional Mutual Information Regularization

This directory contains source code for evaluating federated learning with different methods on various models and tasks. The code was developed for a paper, "FedCR: Personalized Federated Learning Based on Across-Client Common Representation with Conditional Mutual Information Regularization".

## Requirements
 
Some pip packages are required by this library, and may need to be installed. For more details, see `requirements.txt`. We recommend running `pip install --requirement "requirements.txt"`.

Below we give a summary of the datasets, tasks, and models used in this code.


## Task and dataset summary

Note that we put the dataset under the directory .\federated-learning-master\Folder

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

| Directory        | Model                               | Task Summary              |
|------------------|-------------------------------------|---------------------------|
| CIFAR-10         | CNN (with two convolutional layers) | Image classification      |
| CIFAR-100        | CNN (with two convolutional layers) | Image classification      |
| EMNIST           | NN (fully connected neural network) | Digit recognition         |
| FMNIST           | CNN (with two convolutional layer   | Image classification      |

<!-- mdformat on -->


## Training
In this code, we compare 10 optimization methods: **FedAvg**, **FedAvg-FT**, **FedPer**, **LG-FedAvg**, **FedRep**, **FedBABU**, **Ditto**, **FedSR-FT**, **FedPAC**, and **FedCR**. Those methods use vanilla SGD on clients. To recreate our experimental results for each method, for example, for 100 clients and 10% participation rate, on the cifar100 data set with Dirichlet (0.3) split, run those commands for different methods:

**FedAvg** and ***FedAvg-FT**:
```
python /home/zhanghao/code_Fed_VIB/federated-learning-master/federated-learning-master/main_fed.py  --filepath FedAvg.txt --dataset CIFAR100 --method fedavg --lr 0.01 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --beta 0.001 --bs 10 --local_bs 48
```

**FedPer**:
```
python /home/zhanghao/code_Fed_VIB/federated-learning-master/federated-learning-master/main_fed.py  --filepath FedPer.txt --dataset CIFAR100 --method fedper --lr 0.01 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 1 --gpu 0 --epoch 500 --beta 0.001 --bs 10 --local_bs 48
```

**LG-FedAvg**:
```
python /home/zhanghao/code_Fed_VIB/federated-learning-master/federated-learning-master/main_fed.py  --filepath LG-FedAvg.txt --dataset CIFAR100 --method lg --lr 0.01 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --beta 0.001 --bs 10 --local_bs 48
```

**FedRep**:
```
python main_fed.py  --filepath FedRep.txt --dataset CIFAR100 --method fedrep --lr 0.01 --local_ep 10 --local_rep_ep 1 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --beta 0.001 --bs 10 --local_bs 48
```

**FedBABU**:
```
python main_fed.py  --filepath FedBABU.txt --dataset CIFAR100 --method fedbabu --lr 0.01 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --beta 0.001 --bs 10 --local_bs 48
```

**Ditto**:
```
python /main_fed_ditto.py  --filepath Ditto.txt --dataset CIFAR100 --method ditto --mu 0.1 --lr 0.5 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --beta 0.001 --bs 10 --local_bs 48
```

**FedSR-FT**:
```
python main_fedSR.py  --filepath FedSR-FT.txt --dataset CIFAR100 --method fedSR --lr 0.005 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --num_avg_train 1 --num_avg 1 --epoch 500 --dimZ 256 --CMI 0.001 --L2R 0.001 --bs 10 --local_bs 48 --beta2 1
```

**FedPAC**:
```
python main_fed_PAC.py  --filepath FedPAC.txt --dataset CIFAR100 --method fedPAC --lr 0.01 --beta_PAC 1 --local_ep 10 --local_rep_ep 1 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --epoch 500 --dimZ 512 --beta 0.001 --bs 10 --local_bs 48
```

**FedCR**:
```
python main_fed.py  --filepath FedCR.txt --dataset CIFAR100 --method FedCR --lr 0.05 --local_ep 10 --lr_decay 1 --rule Dirichlet --dir_a 0.3 --gpu 0 --num_avg_train 18 --num_avg 18 --epoch 500 --dimZ 512 --beta 0.0001 --bs 10 --local_bs 48 --beta2 1
```



## Other hyperparameters and reproducibility

All other hyperparameters are set by default to the values used in the `Experiment Details` of our Appendix. This includes the batch size, the number of clients per round, the number of client local updates, local learning rate, and model parameter flags. While they can be set for different behavior (such as varying the number of client local updates), they should not be changed if one wishes to reproduce the results from our paper. 

