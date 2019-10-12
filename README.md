Pytorch implementation for abstract reasoning tasks - [What Can Neural Networks Reason About?]

## File Descriptions
All files ending in *generator.py generate a type of dataset for a given task (e.g. subset-sum, Monster trainer, etc):

The tasks described in our experiment section corresponds to the following subtype IDs (IDs are authors' favorite random numbers):

Furthest pair: subtype 0

Maximum value difference: subtype 2

Monster trainer: subtype 15

Subset sum: subtype 12


## Requirements

- Python 2.7
- [numpy](http://www.numpy.org/)
- [pytorch 0.4.0](http://pytorch.org/)

## Usage
For Tasks Furthest pair and Maximum value difference: 

Use
	$ python treasure\_generator.py --K=6 --n\_objects=20 --coord\_size=8 --age\_range=100 --coord\_range=20 
to generate the animal world dataset

and
 	 $ python main.py --model=$MODEL\_NAME --K=6 --n\_objects=25 --coord\_size=8 --age\_range=100 --coord\_range=20 --data=$DATA\_FILE.pickle  --subtype=0/2	  
to train.

For other tasks, the usage is similar and please refer to the documentation in each data generator file and main.py for more details.
