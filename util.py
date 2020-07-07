import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def calc_output_size(args):
    K, n_objects, coord_size, age_range, coord_range, subtype = args.K, args.n_objects, args.coord_size, args.age_range, args.coord_range, args.subtype
    if subtype == 0 or subtype == 1: # Which 2 objects are the farthest/closet? 
        answer_size = n_objects * (n_objects - 1) / 2
    elif subtype == 2: # Range of age/Min age difference
        answer_size = age_range + 1
    elif subtype == 12: # Subset sum
        answer_size = 2
    elif subtype == 15: # Monster trainer
        answer_size = args.max_level * args.map_width * args.map_dim + 1
    return int(answer_size)

def cast_pairs(y_arr, x_arr, mb): #returns yx
    # cast all relations in the form: relation from b -> a: [b's feature, a's feature]
    x_len, y_len = x_arr.size()[1], y_arr.size()[1]

    x_i = y_arr.repeat(1, x_len, 1)
    x_j = torch.unsqueeze(x_arr, 2)
    x_j = x_j.repeat(1, 1, y_len, 1).view(mb, x_len*y_len, -1)
    x_pair = torch.cat((x_i, x_j), dim=2)
        
    return x_pair

def cvt_coord(i):
    return [(i/5-2)/2., (i%5-2)/2.]


def create_dummy(args, n_dummy, feature_size):
    if n_dummy <= 0:
        return 0
    x = torch.FloatTensor(args.batch_size, n_dummy, feature_size)
    if args.cuda:
        x = x.cuda()
    x = Variable(x)
    np_x = np.zeros((args.batch_size, n_dummy, feature_size))
    for i in range(n_dummy):
        np_x[:,i,0:1] = -100 * (i+1)
    x.data.copy_(torch.from_numpy(np_x))

    return x


def create_coord(args, n_objects):
    coord_oi = torch.FloatTensor(args.batch_size, 2)
    coord_oj = torch.FloatTensor(args.batch_size, 2)
    if args.cuda:
        coord_oi = coord_oi.cuda()
        coord_oj = coord_oj.cuda()
    coord_oi = Variable(coord_oi)
    coord_oj = Variable(coord_oj)

    coord_tensor = torch.FloatTensor(args.batch_size, n_objects, 2)
    if args.cuda:
        coord_tensor = coord_tensor.cuda()
    coord_tensor = Variable(coord_tensor)
    np_coord_tensor = np.zeros((args.batch_size, n_objects, 2))
    for i in range(n_objects):
        np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
    coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

    return coord_tensor


''' Flaten x and add coordinate '''
def add_coord(x, coord_tensor):
    mb = x.size()[0]
    n_channels = x.size()[1]
    d = x.size()[2]

    x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
    # add coordinates
    x_flat = torch.cat([x_flat, coord_tensor],2)
    return x_flat

''' Replicate x num times on axis 1'''
def replicate(x, num):
    x = torch.unsqueeze(x, 1)
    x = x.repeat(1,num,1)
    return x


