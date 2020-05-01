# Guide to results folder

## Experiment 1
This is the original version of the model, trained for 2000 epochs. No switching targets, no activation.

## Expriment 2
We added Padcoder(3) to simulate the rodent staying in the same position for 3 timesteps between each new movement. Training for 1000 epochs, no switching targets, no activation.

## Experiment 3
Try dropout=0.5 between LSTM and bottleneck. Training for 1000 epochs, no switching targets, no activation.

## Experiment 4
Transfer Learning. Used the pretrained LSTM of experiment 1, with new target ensembles. Freeze the LSTM weights during backpropagation and only train the last linear layers, for 1000 epochs.

## Experiment 5
Train the original version of the model with tanh activation function, for 1000 epochs.

## Experiment 6
Train the original version of the model with ReLU activation function, for 1000 epochs.

## Experiment 7
Original architecture with hyperparameter search over the head direction cells. We tested out sizes 24, 36 and 48 on top of the original experiment with 12 head cells.

## Experiment 8
Original architecture with hyperparameter search over the place cells. We tested out sizes 32, 64 and 128 on top of the original experiment with 256 PCs.

## Experiment 9
Pretrain the model with 3 switching heads. This is going to be used to test the agnostic hypothesis. Training for 1000 epochs, no activation.

## Experiment 10
Agnosticity test, use the pretrained LSTM of experiment 7 with 1 new target ensemble (not seen during training). Traing only the linear layers for 1000 epochs.

## Experiment 11
Like experiment 6 and add a second ReLU between the LSTM and the bottleneck. Train for 2000 epochs.
