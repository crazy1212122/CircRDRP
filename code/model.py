import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.data import Data
import numpy as np

torch.backends.cudnn.enabled = False
### 3 class in our network, we used GCN_circ and GCN_dis
class GCN_circ(nn.Module):
    def __init__(self, args):
        super(GCN_circ, self).__init__()
        self.args = args
        self.gcn_cir1_f = GCNConv(self.args.fcir, self.args.fcir)
        self.gcn_cir2_f = GCNConv(self.args.fcir, self.args.fcir)

        self.gat_cir1_f = GATConv(self.args.fcir, self.args.fcir,heads=2,concat=False)
        self.gat_cir2_f = GATConv(self.args.fcir, self.args.fcir,heads=1,concat=False)

        self.cnn_cir = nn.Conv2d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fcir,1),
                               stride=1,
                               bias=True)
        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis2_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gat_dis1_f = GATConv(self.args.fdis, self.args.fdis,heads=4,concat=False,edge_dim=1)

        self.cnn_dis = nn.Conv2d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdis,1),
                               stride=1,
                               bias=True)
        torch.manual_seed(1)
        np.random.seed(1)
        self.x_cir = torch.tensor(np.random.random(size=(self.args.circRNA_number, self.args.fcir)), requires_grad=False, dtype=torch.float32).cuda()
        self.x_dis = torch.tensor(np.random.random(size=(self.args.disease_number, self.args.fdis)), requires_grad=False, dtype=torch.float32).cuda()

    def forward(self,data):
        # layer1_data = self.gcn_cir1_f(self.x_cir, data['circ'].edge_index.cuda(), data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda()) + self.gat_cir1_f(self.x_cir,data['circ'].edge_index.cuda(),data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda())
        gat_result =  self.gat_cir1_f(self.x_cir, data['circ'].edge_index.cuda(),data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda())
        cir_layer1_data = self.gcn_cir1_f(self.x_cir, data['circ'].edge_index.cuda(), data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda()) + gat_result
        x_cir_f1 = torch.relu(cir_layer1_data / 2)
        cir_layer2_data = self.gcn_cir2_f(x_cir_f1, data['circ'].edge_index.cuda(), data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda()) + self.gat_cir2_f(x_cir_f1,data['circ'].edge_index.cuda(),data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda())
        x_cir_f2 = torch.relu(cir_layer2_data / 2)

        dis_layer1_data = self.gcn_dis1_f(self.x_dis, data['dis'].edge_index.cuda(), data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda()) + self.gat_dis1_f(self.x_dis, data['dis'].edge_index.cuda(),data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda())
        x_dis_f1 = torch.relu(dis_layer1_data / 2)
        dis_layer2_data = self.gcn_dis2_f(x_dis_f1, data['dis'].edge_index.cuda(), data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda()) + self.gat_dis1_f(x_dis_f1, data['dis'].edge_index.cuda(),data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda())
        x_dis_f2 = torch.relu(dis_layer2_data / 2)

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        X_dis = X_dis.view(1, self.args.gcn_layers, self.args.fdis, -1)

        dis_fea = self.cnn_dis(X_dis)
        dis_fea = dis_fea.view(self.args.out_channels, self.args.disease_number).t()

        X_cir = torch.cat((x_cir_f1,x_cir_f2),1).t()
        X_cir = X_cir.view(1, self.args.gcn_layers, self.args.fcir, -1)

        cir_fea = self.cnn_cir(X_cir)
        cir_fea = cir_fea.view(self.args.out_channels, self.args.circRNA_number).t()

        return cir_fea.mm(dis_fea.t()),cir_fea,dis_fea

class GCN_drug(nn.Module):
    def __init__(self, args):
        super(GCN_drug,self).__init__()
        self.args = args
        self.gcn_drug1_f = GCNConv(self.args.fdrug, self.args.fdrug)
        self.gcn_drug2_f = GCNConv(self.args.fdrug, self.args.fdrug)
        self.gat_drug1_f = GATConv(self.args.fdrug, self.args.fdrug,heads=4,concat=False,edge_dim=1)

        self.cnn_drug = nn.Conv2d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdrug,1),
                               stride=1,
                               bias=True)
        self.gcn_cir1_f = GCNConv(self.args.fcir, self.args.fcir)
        self.gcn_cir2_f = GCNConv(self.args.fcir, self.args.fcir)

        self.gat_cir1_f = GATConv(self.args.fcir, self.args.fcir,heads=1,concat=False)

        self.cnn_cir = nn.Conv2d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fcir,1),
                               stride=1,
                               bias=True)
        torch.manual_seed(1)
        np.random.seed(1)
        self.x_cir = torch.tensor(np.random.random(size=(self.args.circRNA_number, self.args.fcir)), requires_grad=False, dtype=torch.float32).cuda()
        self.x_drug = torch.tensor(np.random.random(size=(self.args.drug_number, self.args.fdrug)), requires_grad=False, dtype=torch.float32).cuda()
        
        # x_drug = torch.randn(self.args.drug_number, self.args.fdrug)
        # x_cir = torch.randn(self.args.circRNA_number, self.args.fcir)
    def forward(self,data):

        drug_layer1_data = self.gcn_drug1_f(self.x_drug, data['drug'].edge_index.cuda(), data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda()) + self.gat_drug1_f(self.x_drug, data['drug'].edge_index.cuda(),data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda())
        x_drug_f1 = torch.relu(drug_layer1_data / 2)
        drug_layer2_data = self.gcn_drug2_f(x_drug_f1, data['drug'].edge_index.cuda(), data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda()) + self.gat_drug1_f(x_drug_f1, data['drug'].edge_index.cuda(),data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda())
        x_drug_f2 = torch.relu(drug_layer2_data / 2)

        gat_result1 = self.gat_cir1_f(self.x_cir,data['circ'].edge_index.cuda(),data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda())
        cir_layer1_data = self.gcn_cir1_f(self.x_cir, data['circ'].edge_index.cuda(), data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda()) + gat_result1
        x_cir_f1 = torch.relu(cir_layer1_data / 2)
        cir_layer2_data = self.gcn_cir2_f(x_cir_f1, data['circ'].edge_index.cuda(), data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda()) + self.gat_cir1_f(x_cir_f1,data['circ'].edge_index.cuda(),data['circ'].x[data['circ'].edge_index[0], data['circ'].edge_index[1]].cuda())
        x_cir_f2 = torch.relu(cir_layer2_data / 2)

        X_drug = torch.cat((x_drug_f1, x_drug_f2), 1).t()
        X_drug = X_drug.view(1, self.args.gcn_layers, self.args.fdrug, -1)

        drug_fea = self.cnn_drug(X_drug)
        drug_fea = drug_fea.view(self.args.out_channels, self.args.drug_number).t()

        X_cir = torch.cat((x_cir_f1,x_cir_f2),1).t()
        X_cir = X_cir.view(1, self.args.gcn_layers, self.args.fcir, -1)

        cir_fea = self.cnn_cir(X_cir)
        cir_fea = cir_fea.view(self.args.out_channels, self.args.circRNA_number).t()

        return cir_fea.mm(drug_fea.t()),drug_fea

class GCN_dis(nn.Module):
    def __init__(self,args):
        super(GCN_dis,self).__init__()
        self.args = args
        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis2_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gat_dis1_f = GATConv(self.args.fdis, self.args.fdis,heads=4,concat=False,edge_dim=1)

        self.cnn_dis = nn.Conv2d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdis,1),
                               stride=1,
                               bias=True)
        self.gcn_drug1_f = GCNConv(self.args.fdrug, self.args.fdrug)
        self.gcn_drug2_f = GCNConv(self.args.fdrug, self.args.fdrug)
        self.gat_drug1_f = GATConv(self.args.fdrug, self.args.fdrug,heads=4,concat=False,edge_dim=1)

        self.cnn_drug = nn.Conv2d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdrug,1),
                               stride=1,
                               bias=True)
    def forward(self,data):
        torch.manual_seed(1)
        x_dis = torch.randn(self.args.disease_number, self.args.fdis)
        x_drug = torch.randn(self.args.drug_number, self.args.fdrug)
        
        dis_layer1_data = self.gcn_dis1_f(x_dis.cuda(), data['dis'].edge_index.cuda(), data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda()) + self.gat_dis1_f(x_dis.cuda(), data['dis'].edge_index.cuda(),data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda())
        x_dis_f1 = torch.relu(dis_layer1_data / 2)
        dis_layer2_data = self.gcn_dis2_f(x_dis_f1, data['dis'].edge_index.cuda(), data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda()) + self.gat_dis1_f(x_dis_f1, data['dis'].edge_index.cuda(),data['dis'].x[data['dis'].edge_index[0], data['dis'].edge_index[1]].cuda())
        x_dis_f2 = torch.relu(dis_layer2_data / 2)

        drug_layer1_data = self.gcn_drug1_f(x_drug.cuda(), data['drug'].edge_index.cuda(), data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda()) + self.gat_drug1_f(x_drug.cuda(), data['drug'].edge_index.cuda(),data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda())
        x_drug_f1 = torch.relu(drug_layer1_data / 2)
        drug_layer2_data = self.gcn_drug2_f(x_drug_f1, data['drug'].edge_index.cuda(), data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda()) + self.gat_drug1_f(x_drug_f1, data['drug'].edge_index.cuda(),data['drug'].x[data['drug'].edge_index[0], data['drug'].edge_index[1]].cuda())
        x_drug_f2 = torch.relu(drug_layer2_data / 2)

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        X_dis = X_dis.view(1, self.args.gcn_layers, self.args.fdis, -1)

        dis_fea = self.cnn_dis(X_dis)
        dis_fea = dis_fea.view(self.args.out_channels, self.args.disease_number).t()

        X_drug = torch.cat((x_drug_f1, x_drug_f2), 1).t()
        X_drug = X_drug.view(1, self.args.gcn_layers, self.args.fdrug, -1)

        drug_fea = self.cnn_drug(X_drug)
        drug_fea = drug_fea.view(self.args.out_channels, self.args.drug_number).t()

        return drug_fea.mm(dis_fea.t()),dis_fea,drug_fea
        
