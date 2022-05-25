The model is built up on the base-line architecture provided. The code can be run using the following steps:

1. Importing all necessary libraries
2. Defining two Dataset functions: one for the training and validation set, which contains features and labels and the other for test set, which only contains features.\
3. Defining data loaders for all the training, validation and test data sets, in order to load the data files, and pre-process them appropriately.\
4. Defining the model:\
   Here, I first started with the baseline model and then tried modifying it to get better accuracy. Firstly, I added more Linear layers, which increased the accuracy. The number of epochs used were 10, however,\
the accuracy kept dropping after epoch number 4. Then I added some more layers of size 1024, which again increased the accuracy. Tuning the learning rate did not give any good results, so kept it constant. Further, I added BatchNorm layers to all the hidden layers\'97this significantly increased the accuracy. I tuned these, by adding and deleting the BatchNorm layers after each hidden layer. Finally stopped when the accuracy reached around 75% on the validation set. Also tried adding the dropout layers but it somehow did not improve the results. I also tuned the batch size, for the training set and the test set as well, increasing the batch size improved the accuracy to some degree. The point to notice was that after 8 epochs or so, the accuracy dropped significantly on the test set (went from 75%-73%), so I kept the number of epochs to 5.\
5. Training the model, defined in the previous step on the training set\
6. Performing validation on the validation set provided\
7. Performing tests on the test set and saving the model and test results.}