"""Generator for a simple RPG game.

There are N monsters throughout a map, with level 1, 2, 3, ..., N.
The player start at a random position with level 0.
Defeating a monster takes the cost of (Monster LV-Player LV) * travel distance.
After defeating each monster, the player will level up to the monster's level.
We ask the model to predict the minimum cost of reaching a target level K.
"""
from argparse import ArgumentParser
import numpy as np
import pickle
from scipy.sparse.csgraph import shortest_path


def build_dataset(n_examples, args):
    """Sample weighted graphs and compute shortest path."""
    # Sample dataset s.t. there are equal number of solutions for each n_stop.
    # For example, if min_stop = 1 and max_stop = 3, we sample equal number of
    # examples whose shortest path have 1, 2, and 3 stops.
    n_stop_types = args.max_stops - args.min_stops + 1
    n_stop_hist = np.zeros(args.max_level + 1, dtype='int64')
    assert n_examples % n_stop_types == 0
    n_examples_per_stop = n_examples // n_stop_types

    dataset = []
    while len(dataset) < n_examples:
        # sample monsters and map
        target_level = np.random.randint(args.max_level) + 1
        has_overlap = True  # sample positions until no duplicate points
        while has_overlap:
            positions = np.random.randint(
                args.map_width + 1,
                size=(args.max_level + 1, args.map_dim)
            )
            has_overlap = False
            for i in range(positions.shape[0]):
                for j in range(i + 1, positions.shape[0]):
                    if np.allclose(positions[i, :], positions[j, :]):
                        has_overlap = True

        levels = np.arange(args.max_level + 1)
        qst = target_level * np.ones(args.max_level + 1)
        data = np.hstack([positions, levels[:, None], qst[:, None]])

        # compute shortest path
        graph = np.zeros((args.max_level + 1, args.max_level + 1))
        for i in range(graph.shape[0]):
            for j in range(i + 1, graph.shape[0]):
                graph[i, j] = (j - i) * np.abs(positions[j, :]
                                               - positions[i, :]).sum()
        distances, predecessors = shortest_path(graph, indices=0,
                                                return_predecessors=True)
        ans = distances[target_level]

        # print()
        # print('Target LV = %d' % target_level)
        # print('Positions: %s' % positions)
        # print('Shortest path distances: %s' % distances)
        # print('Shortest path predecessors: %s' % predecessors)

        # count number of edges on shortest path
        n_stops = 0
        v = target_level
        while v != 0:
            v = predecessors[v]
            n_stops += 1
        if (args.min_stops <= n_stops <= args.max_stops
                and n_stop_hist[n_stops] < n_examples_per_stop):
            n_stop_hist[n_stops] += 1
            np.random.shuffle(data)
            dataset.append((data, ans))
    print('histogram of shortest path length: %s' % list(n_stop_hist))
    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', default='',
                        help='path to save dataset')
    parser.add_argument('--max_level', default=10, type=int)
    parser.add_argument('--map_width', default=10, type=int)
    parser.add_argument('--map_dim', default=2, type=int)
    # CAVEAT: we assume these are divisible by max_level
    parser.add_argument('--n_train', default=60000, type=int,
                        help='train set size')
    parser.add_argument('--n_dev', default=6000, type=int,
                        help='validation set size')
    parser.add_argument('--n_test', default=6000, type=int,
                        help='test set size')
    parser.add_argument('--min_stops', default=3, type=int,
                        help='number of min stops in solution')
    parser.add_argument('--max_stops', default=7, type=int,
                        help='number of max stops in solution')
    parser.add_argument('--repeat', default=0, type=int,
                        help='number of repeats for faster data generation')
    args = parser.parse_args()

    print('building train dataset...')
    train = build_dataset(args.n_train, args)
    print('building dev dataset...')
    dev = build_dataset(args.n_dev, args)
    print('building test dataset...')
    test = build_dataset(args.n_test, args)

    output = 'data/rpg_%s_N%d_Width%d_Dim%d_Train%d_Dev%d_Test%d_%d.pkl' % (
        args.output, args.max_level, args.map_width, args.map_dim,
        args.n_train, args.n_dev, args.n_test, args.repeat
    )
    with open(output, 'wb') as f:
        pickle.dump((train, test, dev), f)

    print("data saved to %s" % output)


if __name__ == '__main__':
    main()
