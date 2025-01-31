# CircRDRP

## Introduction

This is the implementation of our paper"Integrative Graph-Based Framework for Predicting circRNA Drug Resistance Using Disease Contextualization and Deep Learning". It offers a new and effective method in CircRNA and drug resistance  associations'  prediction based on GNN networks. We showed the code of our CircRDRP model.

For details, please see our paper by https://ieeexplore.ieee.org/abstract/document/10670305, which has been accepted by IEEE Journal of Biomedical and Health Informatics.

## Folder

- code : code of CircRDRP model
- dataset : the dataset used by CircRDRP and our experiment
- result: the folder to record the test’s result

## Dataset

In this study, we put forward a new benchmark dataset, which consists of ncRNADrug database, circAtlas, CircFunBase, circBase databases, circBank, DO Ontology, circ2traits,  circR2Disease , circRNADisease and MNDR v3.0 databases. It is put in the folder named *dataset*. Here we simply introduce the dataset's content:

- circ-dis.csv : It is the 0-1 association matrix between the circRNA and diseases in our dataset, and is called $A_{circ-dis}\in R^{1885*30}$ in our paper.
- circ-drug.csv: It is the already proved association matrix between the circRNA and drugs in our dataset and is called $A_{circ-drug}\in R^{1885*27}$ in our paper.
- circname.xlsx: It is the list of all the circRNAs' name we used in our dataset, and they are listed in order.
- circRNA.csv: It is the final circRNAs’ integrated similarity matrix $CS$.
- dis.csv: It is the final integrated similarity matrix of diseases, which was named as $DS$ in our paper.
- disname.xlsx: It includes all the diseases' names utilized in our dataset and they are listed in order.
- drug.csv: It is the final integrated similarity matrix of drugs, which was named as $MS$ in our paper.
- drug-dis.csv: It is the proved 0-1 association matrix between the drugs and diseases in our dataset, and it is recorded as $A_{drug-dis}\in R^{27*30}$.
- drugname.xlsx: It includes all the drugs' names used in our dataset and they are listed in order.

Due to the GitHub's file size limitation, we put our dataset in a .zip file.

As for the process procedure, please see our paper in detail. Meanwhile, in our work, we also utilized the dataset from another paper [[Predicting circRNA-drug resistance associations based on a multimodal graph representation learning framework](https://ieeexplore.ieee.org/abstract/document/10195936/)] and their dataset are also put in *../dataset/Another_dataset* (if you unzip our dataset.zip package).



## Code Structure

- main.py : the main body to start our model and training.
- dataget.py : the code to read dataset and data process steps.
- model.py : the structure of our fused GNN network.
- param.py : the parameter configuration file.
- train.py : the training process code.
- evaluation_scores.py : the code to calculate ML metrics.



## Environment

Our experiment's environment are listed below. 

And our experiment is run on the Nvidia V100 (32GB) GPU device. Besides,  22GB of GPU memory is required.

python==3.8.18

pytorch=1.12.1

torchvision=0.13.1

cudatoolkit=11.3.1

scikit-learn=1.3.0

scipy=1.8.1

tqdm=4.64.1

pyg=2.5.0

pandas=1.4.4

pillow=10.2.0

numpy=1.22.4



## Training and Evaluation

First of all, you should choose the proper dataset and set all the path right. 

Then you can set all the parameters in the param.py or in bash command.

If you want to repeat our experiments, just follow the commands below:

`cd ./code`

`python main.py`



## Citation

```

```

