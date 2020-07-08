Pytorch implementation for abstract reasoning tasks - [What Can Neural Networks Reason About?]

## File Descriptions
All files ending in *generator.py generate a type of dataset for a given task (e.g. subset-sum, Monster trainer, etc):

The tasks described in our experiment section corresponds to the following subtype IDs (IDs are authors' favorite random numbers):

Furthest pair: subtype 0

Maximum value difference: subtype 2

Subset sum: subtype 12

Monster trainer: subtype 15

## Requirements

- Python 2.7
- [numpy](http://www.numpy.org/)
- [pytorch 0.4.0](http://pytorch.org/)

## Usage
- n_iter: number of message passing layers in GNN 
- n\_objects: number of graph nodes in the input dataset 

See appendix G for the set of hyperparameters we use for each task

### For tasks Furthest pair (subtype 0) and Maximum value difference (subtype 2): 
- K: number of animal types (6 by default)
- n\_objects: number of animals (25 by default)

Use
	$ python treasure\_generator.py --K=6 --n\_objects=25 --coord\_size=8 --age\_range=100 --coord\_range=20 
to generate the animal world dataset

and 
	$ python main.py --model=$MODEL\_NAME --lr=$LEARNING_RATE --batch_size=$BATCH_SIZE --hidden_dim=$HIDDEN_DIMENSION --data=$DATA\_FILE.pickle --epochs=$EPOCHS --fc_output_layer=$NUMBER_OF_LAYERS_IN_OUTPUT_FC --mlp_layer=$NUMBER_OF_LAYERS_IN_GNN_AGGREGATION --K=6 --n_iter=3 --n_objects=25 --coord_size=8 --age_range=100 --coord_range=20 --subtype=0 (or --subtype=2 for task Maximum value difference) 
to train.

### For task subset sum:

Use 
	$ python subset_sum_generator.py
to generate the dataset

and 
	$ python main.py --model=$MODEL\_NAME --lr=$LEARNING_RATE --batch_size=$BATCH_SIZE --hidden_dim=$HIDDEN_DIMENSION --data=$DATA\_FILE.pickle --epochs=$EPOCHS --fc_output_layer=$NUMBER_OF_LAYERS_IN_OUTPUT_FC --mlp_layer=$NUMBER_OF_LAYERS_IN_GNN_AGGREGATION --n_iter=5 --n_objects=6 --coord_size=1 --subtype=12

### For task Monster trainer: 
- max_level: number of other objects (except self) in the monster trainer task

Use
	$ python monster_trainer_generator.py --max_level=10
to generate the monster trainer dataset

and
	$ python main.py --model=$MODEL\_NAME --lr=$LEARNING_RATE --batch_size=$BATCH_SIZE --hidden_dim=$HIDDEN_DIMENSION --data=$DATA\_FILE.pickle --epochs=$EPOCHS --fc_output_layer=$NUMBER_OF_LAYERS_IN_OUTPUT_FC --mlp_layer=$NUMBER_OF_LAYERS_IN_GNN_AGGREGATION --aggregate=NEIGHBORHOOD_AGGREGATION_FUNCTION --n_objects=11 --n_iter=7 --subtype=15 
to train
