import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import LSTM
from torch.autograd import Variable
import itertools

from MLPs import *
from conv import *
from util import *

DEFAULT_MODE = -1
DEFAULT_PAIR = (-1,-1)
DEFAULT_IND = -1

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name
        self.return_correct = args.return_correct

    def train_(self, input_nodes, label):
        self.optimizer.zero_grad()

        output = self(input_nodes)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct.to(dtype=torch.float) * 100. / len(label)
        return accuracy, loss
    
    def test_(self, input_nodes, label):
        with torch.no_grad():
            output = self(input_nodes)
            loss = F.nll_loss(output, label)
            pred = output.data.max(1)[1]
            correct_ind = pred.eq(label.data).cpu()
            correct = pred.eq(label.data).cpu().sum()
            accuracy = correct.to(dtype=torch.float) * 100. / len(label)
            if self.return_correct:
                return accuracy, loss, correct_ind, pred.cpu()
            else:
                return accuracy, loss

    def pred_(self, input_nodes):
        with torch.no_grad():
            output = self(input_nodes)
            pred = output.data.max(1)[1]
            return pred

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class GNN(BasicModel):
    def __init__(self, args):
        super(GNN, self).__init__(args, 'GNN')

        def index_shuffle(drop_rate, n_edges):
            if drop_rate <= 0.0:
                return torch.from_numpy(np.arange(n_edges))
            perm = torch.randperm(n_edges)
            k = int(np.floor(n_edges*(1 - drop_rate)))  
            idx = perm[:k]
            return idx
        
        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_object = args.n_objects
        self.subtype = args.subtype
        self.aggregate = args.aggregate
        self.all_same, self.dummy_dir, self.dummy_share = args.all_same, args.dummy_dir, args.dummy_share
        self.cuda_flag = args.cuda
        self.only_max_dummy = args.only_max_dummy
        
        self.answer_size = calc_output_size(args)

        self.n_dummy = args.n_dummy
        self.n_objects = self.n_object + self.n_dummy
        self.n_edges = self.n_objects ** 2
        self.selected_ind = index_shuffle(args.drop_edges, self.n_edges)
        self.num_sind = self.selected_ind.numel()
        self.hidden_dim = args.hidden_dim
        self.mlp_layer = args.mlp_layer
        self.n_iter = args.n_iter
        self.fc_output_layer = args.fc_output_layer
        self.add_features = args.add_features
        
        self.node_vector = torch.FloatTensor(args.batch_size)
        self.pair_vector = torch.FloatTensor(args.batch_size)

        # self.conv = ConvInputModel()
        self.MLP0 = torch.nn.ModuleList()
        if not self.all_same:
            self.MLPdd, self.MLPdios, self.MLPodis = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()

            for i in range(self.n_dummy):
                self.MLPdios.append(torch.nn.ModuleList())
                self.MLPodis.append(torch.nn.ModuleList())
        
        for layer in range(self.n_iter):
            if layer == 0:
                self.MLP0.append(MLP(self.mlp_layer, (self.coord_size + self.add_features)*2, self.hidden_dim, self.hidden_dim))
                
                if not self.all_same:
                    self.MLPdd.append(MLP(self.mlp_layer, (self.coord_size + self.add_features)*2, self.hidden_dim, self.hidden_dim))
                    for i in range(self.n_dummy):
                        self.MLPdios[i].append(MLP(self.mlp_layer, (self.coord_size + self.add_features)*2, self.hidden_dim, self.hidden_dim))
                        self.MLPodis[i].append(MLP(self.mlp_layer, (self.coord_size + self.add_features)*2, self.hidden_dim, self.hidden_dim))
            else:
                self.MLP0.append(MLP(self.mlp_layer, self.hidden_dim*2, self.hidden_dim, self.hidden_dim))
                
                if not self.all_same:
                    self.MLPdd.append(MLP(self.mlp_layer, self.hidden_dim*2, self.hidden_dim, self.hidden_dim))
                    for i in range(self.n_dummy):
                        self.MLPdios[i].append(MLP(self.mlp_layer, self.hidden_dim*2, self.hidden_dim, self.hidden_dim))
                        self.MLPodis[i].append(MLP(self.mlp_layer, self.hidden_dim*2, self.hidden_dim, self.hidden_dim))
        
        if not self.all_same:
            if self.dummy_share:
                for i in range(self.n_dummy):
                    self.MLPdios[i] = self.MLPdios[0]
                    if self.dummy_dir:   
                        self.MLPodis[i] = self.MLPodis[0]
                    else:
                        self.MLPodis[i] = self.MLPdios[0]
            else:
                for i in range(self.n_dummy):
                    if not self.dummy_dir:
                        self.MLPodis[i] = self.MLPdios[i]
            
        self.fcout = FCOutputModel(self.fc_output_layer, self.hidden_dim, self.hidden_dim, self.answer_size)

        self.dummy_nodes = create_dummy(args, self.n_dummy, self.coord_size + self.add_features)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        
    ''' One iteration/layer of GNNs, call n_layer times in forward
    '''
    def reason_step(self, x_flat, layer, mb, step_layer=DEFAULT_MODE, step_type=DEFAULT_MODE, node_pair=DEFAULT_PAIR, node_ind=DEFAULT_IND):
        # cast all pairs against each other
        x_i = x_flat.repeat(1,self.n_objects,1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = x_j.repeat(1,1,self.n_objects,1).view(mb, self.n_objects**2, -1)
        
        x_pair = torch.cat((x_i, x_j), dim=2)
        
        relations = self.MLP0[layer](x_pair)
        relations = relations.view(mb, self.n_objects, self.n_objects, -1)
        
        if layer == step_layer and step_type == 1:
            print("Enter reason_step at layer %d with the pair of node: %s" %(layer, node_pair))
            self.pair_vector = relations[:, node_pair[0], node_pair[1], :]
        
        if self.aggregate == 'max':
            if self.only_max_dummy:
                x_tmp = torch.sum(relations, dim=2)
                x_tmp_dummy, _ = torch.max(relations, dim=2)
                x_tmp_dummy = x_tmp_dummy[:, self.n_object:, :]
                x_tmp[:, self.n_object:, :] = x_tmp_dummy
                x = x_tmp
            else:
                x, _ = torch.max(relations, dim=2)
        else:
            x = torch.sum(relations, dim=2) # Batch, n_objects, relation_dim
            
        if layer == step_layer and step_type == 2:
            print("Enter reason_step at layer %d with query node ind = %d" %(layer, node_ind))
            self.node_vector = x[:, node_ind, :]
            
        return x
    
    def forward(self, x, step_layer=DEFAULT_MODE, step_type=DEFAULT_MODE, node_pair=DEFAULT_PAIR, node_ind=DEFAULT_IND):
        """g"""
        mb = x.size()[0]

        # n_channels = x.size()[1]
        # x = add_coord(x, self.coord_tensor)
        
        if self.n_dummy > 0:
            x = torch.cat([x, self.dummy_nodes], 1)
        
        for layer in range(self.n_iter):
            if self.all_same or self.n_dummy == 0:
                x = self.reason_step(x, layer, mb, step_layer, step_type, node_pair, node_ind)
            else:
                x = self.reason_step_fine_grain(x, layer, mb, step_layer, step_type, node_pair, node_ind)

        x_g = x.sum(1).squeeze()
        
        """f"""
        x = self.fcout(x_g)
        
        return x
        
    def reason_step_fine_grain(self, x_flat, layer, mb, step_layer=DEFAULT_MODE, step_type=DEFAULT_MODE, node_pair=DEFAULT_PAIR, node_ind=DEFAULT_IND):
        x_flat_objects = x_flat[:, :self.n_object, :]
        x_flat_dummy = x_flat[:, self.n_object:, :]
        
        x_pair_oo = cast_pairs(x_flat_objects, x_flat_objects, mb)
        x_pair_dd = cast_pairs(x_flat_dummy, x_flat_dummy, mb)
        
        relations_oo = self.MLP0[layer](x_pair_oo).view(mb, self.n_object, self.n_object, -1)
        relations_dd = self.MLPdd[layer](x_pair_dd).view(mb, self.n_dummy, self.n_dummy, -1)
        
        relations_do, relations_od = [], []
        for i in range(self.n_dummy):
            x_pair_dio = cast_pairs(x_flat_dummy[:, i:i+1, :], x_flat_objects, mb)
            x_pair_odi = cast_pairs(x_flat_objects, x_flat_dummy[:, i:i+1, :], mb)
            
            relations_dio = self.MLPdios[i][layer](x_pair_dio)
            relations_odi = self.MLPodis[i][layer](x_pair_odi)
            
            relations_do.append(relations_dio.unsqueeze(0))
            relations_od.append(relations_odi.unsqueeze(0))
        
        relations_do = torch.cat(relations_do).view(mb, self.n_object, self.n_dummy, -1)
        relations_od = torch.cat(relations_od).view(mb, self.n_dummy, self.n_object, -1)
        
        feature_size = relations_oo.size()[-1]
        relations = torch.FloatTensor(mb, self.n_objects, self.n_objects, feature_size)
        if self.cuda_flag:
            relations = relations.cuda()
        relations[:, :self.n_object, :self.n_object, :] = relations_oo #oo
        relations[:, :self.n_object, self.n_object:, :] = relations_do #do
        relations[:, self.n_object:, :self.n_object, :] = relations_od #od
        relations[:, self.n_object:, self.n_object:, :] = relations_dd #dd
        
        if layer == step_layer and step_type == 1:
            print("Enter reason_step_fg at layer %d with the pair of node: %s" %(layer, node_pair))
            if node_pair[0] >= self.n_object:
                if node_pair[1] >= self.n_object:
                    self.pair_vector = relations_dd[:, node_pair[1]-self.n_object, node_pair[0]-self.n_object, :]
                else:
                    self.pair_vector = relations_do[:, node_pair[1], node_pair[0]-self.n_object, :]
            else:
                if node_pair[1] >= self.n_object:
                    self.pair_vector = relations_od[:, node_pair[1]-self.n_object, node_pair[0], :]
                else:
                    self.pair_vector = relations_oo[:, node_pair[1], node_pair[0], :]

        if self.aggregate == 'max':
            if self.only_max_dummy:
                x_tmp = torch.sum(relations, dim=2)
                x_tmp_dummy, _ = torch.max(relations, dim=2)
                x_tmp_dummy = x_tmp_dummy[:, self.n_object:, :]
                x_tmp[:, self.n_object:, :] = x_tmp_dummy
                x = x_tmp
            else:
                x, _ = torch.max(relations, dim=2)
        else:
            x = torch.sum(relations, dim=2) # Batch, n_objects, relation_dim

        if layer == step_layer and step_type == 2:
            print("Enter reason_step at layer %d with query node ind = %d" %(layer, node_ind))
            self.node_vector = x[:, node_ind, :]
            
        return x

class GNN_R(BasicModel):
    def __init__(self, args):
        super(RRN, self).__init__(args, 'GNN_R')

        def index_shuffle(drop_rate, n_edges):
            if drop_rate <= 0.0:
                return torch.from_numpy(np.arange(n_edges))
            perm = torch.randperm(n_edges)
            k = int(np.floor(n_edges*(1 - drop_rate)))  
            idx = perm[:k]
            return idx
        
        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_objects = args.n_objects
        self.subtype = args.subtype
        self.aggregate = args.aggregate

        self.answer_size = calc_output_size(args)

        self.n_dummy = args.n_dummy
        self.n_objects = self.n_objects + self.n_dummy
        self.n_edges = self.n_objects ** 2
        self.selected_ind = index_shuffle(args.drop_edges, self.n_edges)
        self.num_sind = self.selected_ind.numel()
        self.hidden_dim = args.hidden_dim
        self.node_dim = args.node_dim

        self.n_iter = args.n_iter
        self.mlp_layer = args.mlp_layer
        self.fc_output_layer = args.fc_output_layer
        self.lstm_layer = args.lstm_layer
        self.add_features = args.add_features

        # self.conv = ConvInputModel()
            
        # linear for input -> node dim 
        self.input_transform = nn.Linear(self.coord_size + self.add_features, self.node_dim)
        
        # MLP for relations
        self.relation_input = self.node_dim * 2
        self.relation_MLP = MLP(self.mlp_layer, self.relation_input, self.hidden_dim, self.node_dim)
        
        # MLP for node update
        self.node_input = self.node_dim + (self.coord_size + self.add_features)
        self.node_MLP = MLP(self.mlp_layer, self.node_input, self.hidden_dim, self.hidden_dim)

        # LSTM for node update
        self.node_LSTM = LSTM(self.hidden_dim, self.node_dim, num_layers=self.lstm_layer, batch_first=True)

        # MLP for graph-level readout
        self.fcout = FCOutputModel(self.fc_output_layer, self.node_dim, self.hidden_dim, self.answer_size)

        # self.coord_tensor = create_coord(args, n_objects)
        # self.dummy_nodes = create_dummy(args, self.n_dummy, feature_size+coord_size)
        self.dummy_nodes = create_dummy(args, self.n_dummy, self.coord_size + self.add_features)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)


    ''' initial hidden state for LSTM (h0, c0) '''
    def reset_g(self, mb):
        h = (
            torch.zeros(self.lstm_layer, mb, self.node_dim, requires_grad=True).cuda(),
            torch.zeros(self.lstm_layer, mb, self.node_dim, requires_grad=True).cuda()
            )
        return h

    def reason_step(self, x_flat, h, x_input, layer, mb):
        # cast all pairs against each other
        x_i = x_flat.repeat(1,self.n_objects,1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = x_j.repeat(1,1,self.n_objects,1).view(mb, self.n_objects**2, -1)
        x_pair = torch.cat((x_i, x_j), dim=2)

        relations = self.relation_MLP(x_pair)
        relations = relations.view(mb, self.n_objects, self.n_objects, -1)
        if self.aggregate == 'max':
            x, _ = torch.max(relations, dim=2)
        else:
            x = torch.sum(relations, dim=2) # Batch, n_objects, relation_dim

        x_combine = torch.cat((x_input, x), dim=2) 
        input_lstm = self.node_MLP(x_combine)
        x_lstm, h = self.node_LSTM(input_lstm, h)

        return x_lstm, h

    def forward(self, x_input):
        mb = x_input.size()[0]
        
        h = self.reset_g(mb)    # initial hidden state for LSTM
        if self.n_dummy > 0:
            x_input = torch.cat([x_input, self.dummy_nodes], 1)
        
        x = self.input_transform(x_input) # x is node states

        for layer in range(self.n_iter):
            x, h = self.reason_step(x, h, x_input, layer, mb)

        x_g = x.sum(1).squeeze()
        x = self.fcout(x_g)
        
        return x    
    
def findsubsets(N):
    S = list(range(6))
    holder = []
    for m in range(1, N+1):
        holder.extend(list(itertools.combinations(S, m)))
    
    susbets = []
    for s in holder:
        susbets.append(torch.tensor(list(s)).cuda())
    return susbets 

''' Neural Exhaustive Search
'''
class NES(BasicModel):
    def __init__(self, args):
        super(NES, self).__init__(args, 'NES')
        
        self.coord_size = args.coord_size
        self.n_objects = args.n_objects
        self.subtype = args.subtype
        self.aggregate = args.aggregate
        self.answer_size = calc_output_size(args)

        self.hidden_dim = args.hidden_dim
        self.fc_hidden_dim = args.fc_hidden_dim
        self.mlp_layer = args.mlp_layer
        self.mlp_dim = args.mlp_dim
        self.mlp_before = args.mlp_before
        
        self.fc_output_layer = args.fc_output_layer
        self.lstm_layer = args.lstm_layer
        self.add_features = args.add_features
     
        self.subsets = findsubsets(self.n_objects)
        self.num_subsets = len(self.subsets)
        
        # LSTM for subset sum
        self.subset_LSTM = LSTM(input_size=self.coord_size+self.add_features, hidden_size=self.hidden_dim,  num_layers=self.lstm_layer, batch_first=True)

        # MLP for LSTM
        self.MLP_before = MLP(self.mlp_layer, self.hidden_dim, self.mlp_dim, self.hidden_dim)
        
        # MLP for graph-level readout
        self.fcout = FCOutputModel(self.fc_output_layer, self.hidden_dim, self.fc_hidden_dim, self.answer_size)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)


    ''' initial hidden state for LSTM (h0, c0) '''
    def reset_g(self, mb):
        h = (
            torch.zeros(self.lstm_layer, mb, self.hidden_dim, requires_grad=True).cuda(),
            torch.zeros(self.lstm_layer, mb, self.hidden_dim, requires_grad=True).cuda()
            )
        return h
    
    def forward(self, x_input):
        mb = x_input.size()[0]
        
        # x_input = x_input.reshape(mb, self.n_objects)
        
        x_lstm_holder = torch.zeros(mb, self.num_subsets, self.hidden_dim).cuda()
        
        h_input = self.reset_g(mb) 
        for i in range(self.num_subsets):
            subset = self.subsets[i]
           
            x = torch.index_select(x_input, 1, subset)
            
            _, h = self.subset_LSTM(x, h_input)
            h = h[0].squeeze()
            if self.mlp_before:
                h = self.MLP_before(h)
            
            x_lstm_holder[:, i, :] = h
           
            h_input = self.reset_g(mb)
        
            
        if self.aggregate == 'max':
            x, _ = torch.max(x_lstm_holder, dim=1)
        else:
            x = torch.sum(x_lstm_holder, dim=1) 
        x = self.fcout(x)
        
        return x


''' Recurrent Relation Network 
'''
class GNN2(BasicModel):
    def __init__(self, args):
        super(GNN2, self).__init__(args, 'GNN2')

        def index_shuffle(drop_rate, n_edges):
            if drop_rate <= 0.0:
                return torch.from_numpy(np.arange(n_edges))
            perm = torch.randperm(n_edges)
            k = int(np.floor(n_edges*(1 - drop_rate)))  
            idx = perm[:k]
            return idx
        
        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_objects = args.n_objects
        self.subtype = args.subtype
        self.aggregate = args.aggregate

        self.answer_size = calc_output_size(args)

        self.n_dummy = args.n_dummy
        self.n_objects = self.n_objects + self.n_dummy
        self.n_edges = self.n_objects ** 2
        self.selected_ind = index_shuffle(args.drop_edges, self.n_edges)
        self.num_sind = self.selected_ind.numel()
        self.hidden_dim = args.hidden_dim
        self.node_dim = args.node_dim

        self.n_iter = args.n_iter
        self.mlp_layer = args.mlp_layer
        self.fc_output_layer = args.fc_output_layer
        self.lstm_layer = args.lstm_layer
        self.add_features = args.add_features

        # self.conv = ConvInputModel()
            
        # linear for input -> node dim 
        self.input_transform = nn.Linear(self.coord_size + self.add_features, self.node_dim)
        
        # MLP for relations
        self.relation_input = self.node_dim * 2
        self.relation_MLP = MLP(self.mlp_layer, self.relation_input, self.hidden_dim, self.node_dim)
        
        # MLP for node update
        self.node_input = self.node_dim + (self.coord_size + self.add_features)
        self.node_MLP = MLP(self.mlp_layer, self.node_input, self.hidden_dim, self.hidden_dim)

        # LSTM for node update
        self.node_LSTM = LSTM(self.hidden_dim, self.node_dim, num_layers=self.lstm_layer, batch_first=True)

        # MLP for graph-level readout
        self.fcout = FCOutputModel(self.fc_output_layer, self.node_dim, self.hidden_dim, self.answer_size)

        # self.coord_tensor = create_coord(args, n_objects)
        # self.dummy_nodes = create_dummy(args, self.n_dummy, feature_size+coord_size)
        self.dummy_nodes = create_dummy(args, self.n_dummy, self.coord_size + self.add_features)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)


    ''' initial hidden state for LSTM (h0, c0) '''
    def reset_g(self, mb):
        h = (
            torch.zeros(self.lstm_layer, mb, self.node_dim, requires_grad=True).cuda(),
            torch.zeros(self.lstm_layer, mb, self.node_dim, requires_grad=True).cuda()
            )
        return h

    def reason_step(self, x_flat, h, x_input, layer, mb):
        # cast all pairs against each other
        x_i = x_flat.repeat(1,self.n_objects,1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = x_j.repeat(1,1,self.n_objects,1).view(mb, self.n_objects**2, -1)
        x_pair = torch.cat((x_i, x_j), dim=2)

        relations = self.relation_MLP(x_pair)
        relations = relations.view(mb, self.n_objects, self.n_objects, -1)
        if self.aggregate == 'max':
            x, _ = torch.max(relations, dim=2)
        else:
            x = torch.sum(relations, dim=2) # Batch, n_objects, relation_dim

        x_combine = torch.cat((x_input, x), dim=2) 
        input_lstm = self.node_MLP(x_combine)
        x_lstm, h = self.node_LSTM(input_lstm, h)

        return x_lstm, h

    def forward(self, x_input):
        mb = x_input.size()[0]
        
        h = self.reset_g(mb)    # initial hidden state for LSTM
        if self.n_dummy > 0:
            x_input = torch.cat([x_input, self.dummy_nodes], 1)
        
        x = self.input_transform(x_input) # x is node states

        for layer in range(self.n_iter):
            x, h = self.reason_step(x, h, x_input, layer, mb)

        x_g = x.sum(1).squeeze()
        x = self.fcout(x_g)
        
        return x

''' Hypergraph Relation Network 
'''
class HRN(BasicModel):
    def __init__(self, args):
        super(HRN, self).__init__(args, 'HRN')
        
        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_objects = args.n_objects
        self.subtype = args.subtype
        self.aggregate = args.aggregate
        self.answer_size = calc_output_size(args)

        self.n_dummy = args.n_dummy
        self.n_objects = self.n_objects+ self.n_dummy
        self.n_edges = self.n_objects ** 3

        self.hidden_dim = args.hidden_dim
        self.mlp_layer = args.mlp_layer
        self.n_iter = args.n_iter
        self.fc_output_layer = args.fc_output_layer
        self.aggregate = args.aggregate
        self.add_features = args.add_features
        
        self.input_size = self.coord_size + self.add_features

        ##(number of filters per object+coordinate of object)*2+question vector
        #self.conv = ConvInputModel()
        self.MLPs = torch.nn.ModuleList()
        for layer in range(self.n_iter):
            if layer == 0:
                self.MLPs.append(MLP(self.mlp_layer, self.input_size*3, self.hidden_dim, self.hidden_dim))
            else:
                self.MLPs.append(MLP(self.mlp_layer, self.hidden_dim*3, self.hidden_dim, self.hidden_dim))
        
        self.fcout = FCOutputModel(self.fc_output_layer, self.hidden_dim, self.hidden_dim, self.answer_size)

        #self.coord_tensor = create_coord(args, n_objects)
        self.dummy_nodes = create_dummy(args, self.n_dummy, self.input_size)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)


    ''' One iteration/layer of HRN, call n_layer times in forward
    '''
    def reason_step(self, x_flat, layer, mb):
        # cast all triples against each other
        x_i = x_flat.repeat(1,self.n_objects**2,1)   #(mb, n_objects**3, feature) e.g. abcabc...
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = x_j.repeat(1,1,self.n_objects,1).view(mb,self.n_objects**2,-1)
        x_j = x_j.repeat(1, self.n_objects, 1)    #(mb, n_objects**3, feature) e.g. aaabbbccc aaabbbccc...
        x_k = torch.unsqueeze(x_flat, 2)
        x_k = x_k.repeat(1,1,self.n_objects**2,1).view(mb, self.n_objects**3, -1) #(mb, n_objects**3, feature) e.g. a*n_objects^2, b*n_objects^2...

        x_triple = torch.cat((x_i, x_j, x_k), dim=2)   #(mb, n_objects **3, feature*3) e.g. aaabbbccc aaabbbccc...

        relations = self.MLPs[layer](x_triple)   
        relations = relations.view(mb, self.n_objects, self.n_objects ** 2, -1)
        
        # x[i] contains the sum of the hypergraph messages/relations incoming to node i
        if self.aggregate == 'max':
            x, _ = torch.max(relations, dim=2)
        else:
            x = torch.sum(relations, dim=2) # Batch, n_objects, relation_dim

        return x


    def forward(self, x):        
        mb = x.size()[0]
        n_channels = x.size()[1]

        if self.n_dummy > 0:
            x = torch.cat([x, self.dummy_nodes], 1)
        
        for layer in range(self.n_iter):
            x = self.reason_step(x, layer, mb)
        
        x_g = x.sum(1).squeeze()
        
        x = self.fcout(x_g)
        
        return x


'''  Concatenate all object representations'''
class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_objects = args.n_objects
        self.subtype = args.subtype
        self.answer_size = calc_output_size(args)
        self.add_features = args.add_features
        self.sort_by_age = args.sort_by_age
        self.input_size = self.coord_size + self.add_features

        self.hidden_dim = args.hidden_dim
        self.fc_output_layer = args.fc_output_layer
        self.fcout = FCOutputModel(self.fc_output_layer, self.n_objects * self.input_size, self.hidden_dim, self.answer_size) 
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        
    def forward(self, x_input):
        mb = x_input.size()[0]
        #x_out = torch.zeros([mb, self.answer_size]).cuda()
        # for iter in range(30):
        #     index = torch.randperm(self.n_objects).unsqueeze(0)
        #     index = index.repeat(mb, 1).cuda()
        #     x = torch.cat([torch.index_select(a,0,i).unsqueeze(0) for a, i in zip(x_input, index)])
        #     x = x.view(x.size(0), -1)
        #     x = self.fcout(x)

        #     x_out = x_out + x
        # sort x by age
        if self.sort_by_age:
            age = x_input[:,:,self.coord_size] #Sort; x_input[:,:,8]; [:,:,0] original, random laping 
            _, index = age.sort(dim=1)
            x = torch.cat([torch.index_select(a,0,i).unsqueeze(0) for a, i in zip(x_input, index)])
        else:
            x = x_input
        
        x = x.view(x.size(0), -1)
        x_out = self.fcout(x)
        
        return x_out

'''  MLP with skip connections'''
class Skip_MLP(BasicModel):
    def __init__(self, args):
        super(Skip_MLP, self).__init__(args, 'SkipMLP')

        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_objects = args.n_objects
        self.subtype = args.subtype
        self.answer_size = calc_output_size(args)
        self.add_features = args.add_features
        self.sort_by_age = args.sort_by_age
        self.input_size = self.coord_size + self.add_features

        self.hidden_dim = args.hidden_dim
        self.fc_output_layer = args.fc_output_layer
        self.fcout = FCOutputModel_SkipConnection(self.fc_output_layer, self.n_objects * self.input_size, self.hidden_dim, self.answer_size) 
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        
    def forward(self, x_input):
        mb = x_input.size()[0]
        #x_out = torch.zeros([mb, self.answer_size]).cuda()
        # for iter in range(30):
        #     index = torch.randperm(self.n_objects).unsqueeze(0)
        #     index = index.repeat(mb, 1).cuda()
        #     x = torch.cat([torch.index_select(a,0,i).unsqueeze(0) for a, i in zip(x_input, index)])
        #     x = x.view(x.size(0), -1)
        #     x = self.fcout(x)

        #     x_out = x_out + x
        # sort x by age
        if self.sort_by_age:
            age = x_input[:,:,self.coord_size] #Sort; x_input[:,:,8]; [:,:,0] original, random laping 
            _, index = age.sort(dim=1)
            x = torch.cat([torch.index_select(a,0,i).unsqueeze(0) for a, i in zip(x_input, index)])
        else:
            x = x_input
        
        x = x.view(x.size(0), -1)
        x_out = self.fcout(x)
        
        return x_out


''' Deep Sets
Encode position feature (rather than order by position as in MLP), apply order-invariant model deep set 
'''
class DeepSet(BasicModel):
    def __init__(self, args):
        super(DeepSet, self).__init__(args, 'DeepSet')

        self.coord_size = args.coord_size
        self.K, self.age_range, self.coord_range = args.K, args.age_range, args.coord_range
        self.n_objects = args.n_objects
        self.subtype = args.subtype

        self.add_features = args.add_features
        self.answer_size = calc_output_size(args)
            
        self.input_size = self.coord_size + self.add_features

        self.hidden_dim = args.hidden_dim
        self.mlp_layer = args.mlp_layer
        self.fc_output_layer = args.fc_output_layer

        self.MLP = MLP(self.mlp_layer, self.input_size, self.hidden_dim, self.hidden_dim)
        self.fcout = FCOutputModel(self.fc_output_layer, self.hidden_dim, self.hidden_dim, self.answer_size)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)

    def forward(self, x):
        mb = x.size()[0]
        n_channels = x.size()[1]

        x_ = self.MLP(x)

        x_g = x_.view(mb, self.n_objects, -1)
        x_g = x_g.sum(1).squeeze()

        """f"""
        x = self.fcout(x_g)

        return x
