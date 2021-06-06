# Task 1
Given is a training set that consists of the time, fluid temperature and solid temperature.
The goal is to make predictions for the fluid and solid temperature given a testing set that consists of time samples.

## Problems overcome
Build the whole neural network pipeline from scratch:
- read in training and testing data, preprocess it and transform it to the desired network input type
- create predictions using the testing data and write out the result in the correct format
- calculate loss function correctly such that varying batch size is taken into account
- create an adaptable neural network (mainly adapted from provided course code template)
- add progress bar to get a better overview of training process
- save and load model
- IO to keep track of the parameters and losses of each model
- create package for pipeline
- add LICENSE
- create environment.yml 

- train two neural networks (one for the solid temperature and one for the fluid temperature)

## Best Model
![Result](result_1.png)
