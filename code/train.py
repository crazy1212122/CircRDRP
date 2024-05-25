from param import parameter_parser
import torch
from torch_geometric.data import Data


####train1 function is used to measure the loss and optim parameters in the network
def train1(model_circ,model_dis, train_data, optimizer, opt):
    model_circ.train()

    for epoch in range(0, opt.epoch):
        model_circ.zero_grad()
        circ_dis,cir_fea,dis_fea = model_circ(train_data)
        loss1 = torch.nn.BCEWithLogitsLoss(reduction='mean')
        #print(train_data['drug_dis'].shape)
        loss1 = loss1(circ_dis, train_data['c_dis'].x.cuda())
        loss1.backward()
        optimizer.step()
        print("cricRNA-disease loss \t",str(epoch),"\t",loss1.item())
    
    model_dis.train()
    optimizer2 = torch.optim.Adam(model_dis.parameters(),lr=0.005)
    for epoch in range(0, opt.epoch):
        model_dis.zero_grad()
        drug_dis,dis_fea,drug_fea = model_dis(train_data)
        loss3 = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss3 = loss3(drug_dis, train_data['drug_dis'].x.cuda())
        loss3.backward()
        optimizer2.step()
        print("drug-disease loss \t",str(epoch),"\t",loss3.item())
    return model_circ,model_dis
    
##### another training option by divide function train1 into train1 and train3
def train3(model_dis,train_data, optimizer2,opt):
    model_dis.train()

    for epoch in range(0, opt.epoch):
        model_dis.zero_grad()
        drug_dis,dis_fea,drug_fea = model_dis(train_data)
        loss3 = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss3 = loss3(drug_dis, train_data['drug_dis'].x.cuda())
        loss3.backward()
        optimizer2.step()
        print("drug-disease loss \t",str(epoch),"\t",loss3.item())
    return model_dis


