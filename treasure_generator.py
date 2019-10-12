import argparse
import os
import numpy as np
from math import exp
import random
import pickle
from sklearn.neighbors import NearestNeighbors

np.random.seed(1)
random.seed(1)

question_subtypes = 8
train_size = 50000 #Task0: 60000, Task2: 50000
test_size = 5000
validate_size = 5000

dirs = './data'

parser = argparse.ArgumentParser(description='PyTorch Animal World Example')
parser.add_argument('--prefix', type=str, default='', help='dataset filename')
parser.add_argument('--K', type=int, default=6, metavar='K', help='number of animal types')
parser.add_argument('--n_objects', type=int, default=25, metavar='n_objects', help='number of animals')
parser.add_argument('--coord_size', type=int, default=8, metavar='coord_size', help='number of dimensions')
parser.add_argument('--age_range', type=int, default=100, metavar='age_range', help='the largest age of an animal (min is 0)')
parser.add_argument('--coord_range', type=int, default=20, metavar='coord_range', help='00 means range (-10, 10) for each coord')
parser.add_argument('--subtype', type=int, default=0, metavar='subtype', help='question subtypes we want to test')
parser.add_argument('--label_cr', type=float, default=0.0, metavar='label_cr', help='label corruption rate')

args = parser.parse_args()
argument = args.prefix

def build_dataset(args):
    K, n_objects, coord_size, age_range, coord_range, subtype = args.K, args.n_objects, args.coord_size, args.age_range, args.coord_range, args.subtype
    
    data = []
    for i in range(n_objects):
        coord = np.random.choice(coord_range+1, coord_size)
        data.append(coord)
    ages = np.random.choice(age_range+1, n_objects)
    types = np.random.choice(K, n_objects)
    
    answers = calculate_answer(args, data, ages, types)
    
    datat = np.asarray(data).transpose()
    data = np.append(datat, [ages, types], axis=0).transpose()
    
    dataset = (data, answers)
    return dataset
    
def calc_output_range(args):
    K, n_objects, coord_size, age_range, coord_range = args.K, args.n_objects, args.coord_size, args.age_range, args.coord_range
    
    answer_range = np.zeros(question_subtypes, dtype=int)
    for subtype in range(question_subtypes):
        if subtype == 0 or subtype == 1: # Which 2 animals are the farthest/closet? 
            answer_size = K * (K - 1) / 2
        else: # subtype=2 Range of age/Min age difference
            answer_size = age_range + 1
        answer_range[subtype] = answer_size
    return answer_range
    
    
def calculate_answer(args, data, ages, types):
    K, n_objects, coord_size, age_range, coord_range, subtype = args.K, args.n_objects, args.coord_size, args.age_range, args.coord_range, args.subtype
    
    answers = np.zeros(question_subtypes, dtype=int)
    
    if random.random() < args.label_cr:
        answer_range = calc_output_range(args)
        i = 0
        for k in answer_range:
            answers[i] = random.randint(0, k-1)
            i = i+1
        return answers
    
    answers = np.zeros(question_subtypes, dtype=int)
    
    nbrs = NearestNeighbors(n_neighbors=n_objects, p=1).fit(data)
    distances, indices = nbrs.kneighbors(data)

    ele_max = np.argmax(distances[:,-1])
    
    furthest_pair = sorted([types[indices[ele_max][0]], types[indices[ele_max][-1]]])
    answers[0] = pair_2_ind(furthest_pair[0], furthest_pair[1], K) #Which 2 animals are the farthest? 
    
    ele_min = np.argmin(distances[:,1])
    closest_pair = sorted(indices[ele_min][:2])
    answers[1] = pair_2_ind(types[closest_pair[0]], types[closest_pair[1]], K) #Which 2 animals are the Closest?
    
    answers[2] = max(ages) - min(ages) #Range of age
    return answers

def pair_2_ind(i, j, K):
    ind = 0
    for a in range(i):
        ind += K-a
    ind = ind + j - i + 1
    return ind 

K, n_objects, coord_size, age_range, coord_range = args.K, args.n_objects, args.coord_size, args.age_range, args.coord_range
filename = os.path.join(dirs, 'animal_world_%s_K%d_N%d_D%d_A%d_M%d_cr%.2f.pickle' %(argument, K, n_objects, coord_size, age_range, coord_range, args.label_cr))

print('building train datasets...')
train_datasets = [build_dataset(args) for _ in range(train_size)]

args.label_cr = 0.00

print('building test datasets...')
test_datasets = [build_dataset(args) for _ in range(test_size)]
print('building validation datasets...')
validation_datasets = [build_dataset(args) for _ in range(validate_size)]


print('saving datasets...')

if not os.path.exists(dirs):
        os.makedirs(dirs)

with open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets, validation_datasets), f)
print('datasets saved at {}'.format(filename))
