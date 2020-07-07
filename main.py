"""""""""
Pytorch implementation of "What Can Neural Networks Reason About?"
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os
import pickle
import random
import numpy as np
import shutil
import torch
from torch.autograd import Variable

from model import *
import logging

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

best_prec, best_model_test_acc = 0.0, 0.0
is_best = False
best_epoch = 0


parser = argparse.ArgumentParser(description='PyTorch Animal World Example')

#Model specifications
parser.add_argument('--model', type=str, choices=['GNN', 'NES', 'HRN', 'GNN_R', 'CNN-MLP', 'DeepSet', 'Skip-MLP'], default='GNN', help='resume from model stored')
parser.add_argument('--n_iter', type=int, default=1, help='number of GNN/GNN_R iterations/layers (default: 1)')
parser.add_argument('--mlp_layer', type=int, default=3, help='number of layers for MLPs in GNN/GNN_R/MLP (default: 4)')
parser.add_argument('--hidden_dim', type=int, default=256, help='feature hidden dimension of MLPs (default: 128)')
parser.add_argument('--fc_output_layer', type=int, default=4, help='number of layers for output(softmax) MLP in GNN/GNN2/MLP (default: 3)')
parser.add_argument('--fc_hidden_dim', type=int, default=128, help='hidden dimension of MLPs for fcoutput layer; only implemented for NES (default: 128)')
parser.add_argument('--node_dim', type=int, default=64, help='node/lstm feature dimension (default: 64)')
parser.add_argument('--mlp_before', action='store_true', default=False, help='add MLP before MAX')
parser.add_argument('--lstm_layer', type=int, default=1, help='number of layers for LSTM in GNN_R (default: 1)')
parser.add_argument('--mlp_dim', type=int, default=128, help='MLP dimension for BF-LSTM (default: 128)')
parser.add_argument('--aggregate', type=str, default="sum", choices=["sum", "max"], help='aggregate relations: sum or max')
parser.add_argument('--drop_edges', type=float, default=0.0, help='randomly drop drop_edges percent edges, default 0')
parser.add_argument('--n_dummy', type=int, default=0, help='number of dummy nodes for GNNs (default: 0)')
parser.add_argument('--all_same', action='store_true', default=False, help='use one MLP for all relations')
parser.add_argument('--dummy_dir', action='store_true', default=False, help='use different MLP for relation a-b and b-a')
parser.add_argument('--dummy_share', action='store_true', default=False, help='use the same MLP for all dummy nodes')
parser.add_argument('--only_max_dummy', action='store_true', default=False, help='Only doing max when we are aggregating dummy nodes')
parser.add_argument('--sort_by_age', action='store_true', default=False, help='For CNN-MLP, sort by age for the input')

# Training settings
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--resume', type=str, help='resume from model stored')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0001)')
parser.add_argument('--decay', type=float, default=1e-5, help='weight decay (default: 0.0)')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

# Logging and storage settings
parser.add_argument('--log_file', type=str, default='accuracy.log', help='dataset filename')
parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--filename', type=str, default='', help='the file which store trained model')
parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models')

# Data settings
parser.add_argument('--data', type=str, default='animal_world__K6_N25_D5_A100_M10.pickle', help='dataset filename')
parser.add_argument('--no_coord', action='store_true', default=False, help='disables coordinate embedding in object representation')
parser.add_argument('--prefix', type=str, default='', help='dataset filename')
parser.add_argument('--K', type=int, default=6, help='number of animal types')
parser.add_argument('--n_objects', type=int, default=25, help='number of animals')
parser.add_argument('--coord_size', type=int, default=8, help='number of dimensions')
parser.add_argument('--add_features', type=int, default=2, help='number of additional features')
parser.add_argument('--age_range', type=int, default=100, help='the largest age of an animal (min is 0)')
parser.add_argument('--coord_range', type=int, default=20, help='10 means range (-5, 5) for each coord')
parser.add_argument('--subtype', type=int, default=0, help='question subtypes we want to test')
parser.add_argument('--max_weight', default=5, type=int, help='max weight of edge')
parser.add_argument('--max_level', default=5, type=int)
parser.add_argument('--map_width', default=10, type=int)
parser.add_argument('--map_dim', default=2, type=int)

# other settings
parser.add_argument('--return_correct', action='store_true', default=False, help='return correct indices')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.subtype < 10: #0: Maximum value difference and 2: Furthest pair
    args.add_features = 2
elif args.subtype == 12: # Subset sum
    args.add_features = 0

if args.subtype == 15: # Monster trainer
    args.coord_size = args.map_dim + 2
    
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model=='CNN-MLP': 
    model = CNN_MLP(args)
elif args.model=='Skip-MLP':
    model = Skip_MLP(args)
elif args.model=='GNN':
    model = GNN(args)
elif args.model=='DeepSet':
    model = DeepSet(args)
elif args.model=='GNN_R':
    model = GNN_R(args)
elif args.model=='HRN':
    model = HRN(args)
elif args.model=='NES':
    model = NES(args)

scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.5)
                
file_dir = "results"
if not args.no_log:
    files_dir = '%s/%s/Task%s' %(file_dir, args.data, args.subtype)
    args.files_dir = files_dir
    if 'MLP' in args.model:
        args.filename = '%s_sort_%s_%s_%s_%s%s%s_lr%s_%s_%s_bs%s_mlp%s_%s_%s_%s_%s.log' %(args.model, args.sort_by_age, args.n_iter, args.n_dummy, args.all_same, args.dummy_share, args.dummy_dir, args.lr, args.node_dim, args.hidden_dim, args.batch_size, args.mlp_layer, args.fc_output_layer, args.no_coord, args.drop_edges, args.aggregate)
    elif args.model == 'BF-LSTM':
        args.filename = '%s_%s_%s_%s%s%s_lr%s_%s_%s_bs%s_mlp%s_%s_%s_%s_%s_fch_%d_%s_%d.log' %(args.model, args.n_iter, args.n_dummy, args.all_same, args.dummy_share, args.dummy_dir, args.lr, args.node_dim, args.hidden_dim, args.batch_size, args.mlp_layer, args.fc_output_layer, args.no_coord, args.drop_edges, args.aggregate, args.fc_hidden_dim, args.mlp_before, args.mlp_dim)
    else:
        args.filename = '%s_%s_%s_%s%s%s_lr%s_%s_%s_bs%s_mlp%s_%s_%s_%s_%s.log' %(args.model, args.n_iter, args.n_dummy, args.all_same, args.dummy_share, args.dummy_dir, args.lr, args.node_dim, args.hidden_dim, args.batch_size, args.mlp_layer, args.fc_output_layer, args.no_coord, args.drop_edges, args.aggregate)
    
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        datefmt='%m-%d %H:%M',
                        filename="%s/%s" %(args.files_dir, args.filename),
                        filemode='w+')
    
    print(vars(args))
    logging.info(vars(args))
    
subtype = args.subtype
    
model_dirs = './model'
bs = args.batch_size
data_file = args.data

input_nodes = torch.FloatTensor(bs, args.n_objects, args.coord_size+args.add_features)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_nodes = input_nodes.cuda()
    label = label.cuda()

input_nodes = Variable(input_nodes)
label = Variable(label)

def save_checkpoint(state, is_best, epoch, args):
    if not is_best:
        return
    """Saves checkpoint to disk"""
    directory = "models/%s/Task%s/%s/"%(args.data, args.subtype, args.filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint.pth.tar' + '_' + str(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')

def cvt_data_axis(dataset, subtype):
    data = []
    label = []
    
    for d, ans in dataset:
        data.append(d)
        if subtype <= 7:
            label.append(ans[subtype])
        else:
            label.append(ans)
    
    return (data, label)

def tensor_data(data, i):
    nodes = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    
    input_nodes.data.resize_(nodes.size()).copy_(nodes)
    label.data.resize_(ans.size()).copy_(ans)
    
def train(epoch, dataset, subtype):
    model.train()
    train_size = len(dataset)
    
    random.shuffle(dataset)

    data = cvt_data_axis(dataset, subtype)
    
    running_loss = 0.0
    accuracys = []
    losses = []
    for batch_idx in range(train_size // bs):
        tensor_data(data, batch_idx)
        accuracy, loss = model.train_(input_nodes, label)
        running_loss += loss.item()
        
        accuracys.append(accuracy)
        losses.append(loss)
        
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)] Subtype {} accuracy: {:.2f}%, loss: {:.5f}'.format(epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, subtype, accuracy, running_loss/(bs * args.log_interval)))
            logging.info('Train Epoch: {} [{}/{} ({:.2f}%)] Subtype {} accuracy: {:.2f}%, loss: {:.5f}'.format(epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, subtype, accuracy, running_loss/(bs * args.log_interval)))
            running_loss = 0.0
            
    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    print('\nTrain set: Subtype {} accuracy: {:.2f}%, loss: {:.5f}'.format(subtype, avg_accuracy, avg_losses))
    logging.info('\nTrain set: Subtype {} accuracy: {:.2f}%, loss: {:.5f}'.format(subtype, avg_accuracy, avg_losses))

def validate(epoch, dataset, subtype):
    global is_best, best_prec
    
    model.eval()
    test_size = len(dataset)
    data = cvt_data_axis(dataset, subtype)

    accuracys = []
    losses = []
    for batch_idx in range(test_size // bs):
        tensor_data(data, batch_idx)
        accuracy, loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)

    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    print('Validation set: Subtype {} accuracy: {:.2f}%, loss: {:.5f}'.format(subtype, avg_accuracy, avg_losses))
    logging.info('Validation set: Subtype {} accuracy: {:.2f}%, loss: {:.5f}'.format(subtype, avg_accuracy, avg_losses))
    
    is_best = avg_accuracy > best_prec
    best_prec = max(avg_accuracy, best_prec)

def test(epoch, dataset, subtype):
    global is_best, best_model_test_acc, best_epoch
    
    model.eval()
    test_size = len(dataset)
    data = cvt_data_axis(dataset, subtype)

    accuracys = []
    losses = []
    for batch_idx in range(test_size // bs):
        tensor_data(data, batch_idx)
        accuracy, loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)

    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    print('Test set: Subtype {} accuracy: {:.2f}%, loss: {:.5f} \n'.format(subtype, avg_accuracy, avg_losses))
    logging.info('Test set: Subtype {} accuracy: {:.2f}%, loss: {:.5f} \n'.format(subtype, avg_accuracy, avg_losses))
    
    if is_best:
        best_model_test_acc = avg_accuracy
        best_epoch = epoch
        
    if epoch%10 == 0:
        print('************ Best model\'s test accuracy: {:.2f}% (best model is from epoch {}) ************\n'.format(best_model_test_acc, best_epoch))
        logging.info('************ Best model\'s test accuracy: {:.2f}% (best model is from epoch {}) ************\n'.format(best_model_test_acc, best_epoch))
    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs, data_file)
    with open(filename, 'rb') as f:
        train_datasets, test_datasets, validation_datasets = pickle.load(f)

    return train_datasets, test_datasets, validation_datasets
    
train_datasets, test_datasets, validation_datasets = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(1, args.epochs + 1):
    scheduler.step()
    train(epoch, train_datasets, subtype)
    validate(epoch, validation_datasets, subtype)
    test(epoch, test_datasets, subtype)
    if epoch%args.log_interval == 0 and args.save_model:
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
        }, is_best, epoch, args)
        
print('****** Best model\'s test accuracy: {:.2f}% throughout training (best model is from epoch {}) ****** \n'.format(best_model_test_acc, best_epoch))
logging.info('****** Best model\'s test accuracy: {:.2f}% throughout training (best model is from epoch {}) ****** \n'.format(best_model_test_acc, best_epoch))
