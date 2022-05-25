The model is built up on the base-line architecture provided. The code can be run using the following steps:

1. Importing all necessary libraries
2. Defining two Dataset functions: one for the training and validation set, which contains features and labels and the other for test set, which only contains features.\
3. Defining data loaders for all the training, validation and test data sets, in order to load the data files, and pre-process them appropriately.\
4. Defining the model:\
   I started with two CNN, 3 LSTM and 2 linear layers as the initial model, which gave LD of 13 after 20-25 epochs and then saturated, even after changing the lr with a scheduler. Tried manually changing the lr, but the distance only reached upto 13. Then changed the model by adding more layers in the LSTM and managed to get LD of 11 after 25 epochs with an lr scheduler. Final model has 3 CNN layers, 4 LSTM layers with dropouts and 3 linear layers. This gave an LD of ~10/11 after almost 30+ epochs, with lr scheduler as well as manually changing the lr based of the performance in previous epochs. The model again saturated at an LD of 10-11 and kept fluctuating even after increasing the number of layers in the LSTM.
5. Training the model, defined in the previous step on the training set\
6. Performing validation on the validation set provided\
7. Performing tests on the test set and saving the model and test results.}