import csv
import torch
import random
import numpy as np
from torch_geometric.data import Data
  
    
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)
    

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

#####collect the datasets into Data 
def dataset(args):
    dataset = dict()

    c_d_data = read_csv(args.dataset_path + '/circ-drug.csv')
    c_d_edge = get_edge_index(c_d_data)
    dataset['c_d'] = Data(x=c_d_data, edge_index=c_d_edge)

    c_dis_data = read_csv(args.dataset_path + '/circ-dis.csv')
    c_dis_edge = get_edge_index(c_dis_data)
    dataset['c_dis'] = Data(x=c_dis_data,edge_index=c_dis_edge)

    drug_dis_data = read_csv(args.dataset_path + '/drug-dis.csv')
    drug_dis_edge = get_edge_index(drug_dis_data)
    dataset['drug_dis'] = Data(x=drug_dis_data,edge_index=drug_dis_edge)

    zero_index = []
    one_index = []
    cd_pairs = []
    for i in range(dataset['c_d'].x.size(0)):
        for j in range(dataset['c_d'].x.size(1)):
            if dataset['c_d'].x[i][j] < 1 :
                zero_index.append([i, j, 0])
            if dataset['c_d'].x[i][j] >= 1 :
                one_index.append([i, j, 1])
 
    cd_pairs = random.sample(zero_index, len(one_index)) + one_index
    

    drug_matrix = read_csv(args.dataset_path + '/drug.csv')
    drug_edge_index = get_edge_index(drug_matrix)
    dataset['drug'] = Data(x=drug_matrix,edge_index=drug_edge_index)

    cc_matrix = read_csv(args.dataset_path + '/circRNA.csv')
    cc_edge_index = get_edge_index(cc_matrix)
    dataset['circ'] = Data(x=cc_matrix,edge_index=cc_edge_index)

    dis_matrix = read_csv(args.dataset_path + '/dis.csv')
    dis_edge_index = get_edge_index(dis_matrix)
    dataset['dis'] = Data(x=dis_matrix,edge_index=dis_edge_index)

    return dataset, cd_pairs

#####Prepare the features for the classifier
def new_dataset(cir_fea, drug_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []
    
    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
            
        if pair[2] == 0:
            unknown_pairs.append(pair[:2]) 
    print("--------------------")
    print(cir_fea.shape,drug_fea.shape)
    print("--------------------")
    print(len(unknown_pairs), len(known_pairs))
    
    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = cir_fea[unknown_pairs[i][0],:].tolist() + drug_fea[unknown_pairs[i][1],:].tolist()+[0,1]
        nega_list.append(nega)
        
    posi_list = []
    for j in range(len(known_pairs)):
        posi = cir_fea[known_pairs[j][0],:].tolist()+ drug_fea[known_pairs[j][1],:].tolist()+[1,0]
        posi_list.append(posi)
    
    samples = posi_list + nega_list
    
    random.shuffle(samples)
    samples = np.array(samples)
    return samples

def C_Dmatix(cd_pairs,trainindex,testindex):
    c_dmatix = np.zeros((1885,27)) 
    ##If you want to replace our dataset , remenber to change the matrix size here

    for i in trainindex:
        if cd_pairs[i][2]==1:
            c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]]=1
    
    dataset = dict()
    cd_data = []
    cd_data += [[float(i) for i in row] for row in c_dmatix]
    cd_data = torch.Tensor(cd_data)
    dataset['c_drug'] = Data(x=cd_data)

    train_cd_pairs = []
    test_cd_pairs = []
    for m in trainindex:
        train_cd_pairs.append(cd_pairs[m])
    
    for n in testindex:
        test_cd_pairs.append(cd_pairs[n])


    return dataset['c_drug'],train_cd_pairs,test_cd_pairs