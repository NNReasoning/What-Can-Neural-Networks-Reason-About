import argparse
import os
import numpy as np
from math import exp
import random
import pickle

np.random.seed(1)
random.seed(1)

question_subtypes = 12
train_size = 40000
test_size = 4000
validate_size = 4000

dirs = './data'

parser = argparse.ArgumentParser(description='PyTorch Animal World Subset Sum Example')
parser.add_argument('--prefix', type=str, default='', help='dataset filename')
parser.add_argument('--num_range', type=int, default=200, metavar='R', help='range of all numbers are (-R, R)')
parser.add_argument('--n_objects', type=int, default=6, metavar='n_objects', help='number of numbers in a set')
parser.add_argument('--ssum', type=int, default=0, metavar='sum', help='the desired subset sum')
parser.add_argument('--subtype', type=int, default=12, metavar='subtype', help='question subtypes we want to test')
# parser.add_argument('--label_cr', type=float, default=0.0, metavar='subtype', help='question subtypes we want to test')

args = parser.parse_args()
argument = args.prefix

def subset(array, num):
    result = []
    def find(arr, num, path=()):
        if not arr.any():
            return
        if arr[0] == num:
            result.append(path + (arr[0],))
        else:
            find(arr[1:], num - arr[0], path + (arr[0],))
            find(arr[1:], num, path)
    find(array, num)
    return result

def trivial_solution(sub, n):
    for s in sub:
        if len(s) == 1:
            return 1
        elif len(s) == 2:
            return 2
        elif len(s) == n:
            return n
    return 0

def build_dataset(examples):
    num_range, n_objects, ssum, subtype = args.num_range, args.n_objects, args.ssum, args.subtype
    count = examples/2

    true, false = 0, 0
    dataset = []
    while True:
        data = np.random.choice(num_range*2+1, n_objects) - num_range
        sub = subset(data, ssum)

        if len(sub) > 0:
            length = trivial_solution(sub, n_objects)
            if length == 1:
                if random.random() > 0.1:
                    continue
            elif length == 2 or length == n_objects:
                if random.random() > 0.05:
                    continue
            ans = 1
        else:
            ans = 0
        
        data = data.reshape(n_objects, 1)
        if ans and true < count:
            dataset.append((data, ans))
            true += 1
        elif (not ans) and false < count:
            dataset.append((data, ans))
            false += 1
        
        if true == count and false == count:
            break
    random.shuffle(dataset)
    return dataset
    
num_range, n_objects, ssum, subtype = args.num_range, args.n_objects, args.ssum, args.subtype
filename = os.path.join(dirs, 'subset-sum_%s_R%d_N%d_S%d_train_%d.pickle' %(argument, num_range, n_objects, ssum, train_size))

print('building train datasets...')
train_datasets = build_dataset(train_size)

print('building test datasets...')
test_datasets = build_dataset(test_size)

print('building validation datasets...')
validation_datasets = build_dataset(validate_size)

print('saving datasets...')
if not os.path.exists(dirs):
        os.makedirs(dirs)

with open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets, validation_datasets), f)
print('datasets saved at {}'.format(filename))