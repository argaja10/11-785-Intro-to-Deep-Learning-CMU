The code can be run using the following steps:

1. Importing all necessary libraries

2. Loading datasets with ImageFolder and different transformations. Tried a variety of combinations, including normalisation, affine, reshape, horizontal flips, Color jitter, rotation. Too many transformations led to extremely low accuracy in the 0th epoch and didn't reach 1% until 5th epoch. Finally, kept only normalise, horizontal flips, and random affine. Tried with different parameter values for the transformations as well.

3. Defining data loaders for all the training and validation sets.

4. Defining the model and training:
   Tried ResNet16,34 and 50, changed stride for the convolutional layers, tried different weight decay values (5e-5, 5e-4), different learning rates (0.1,0.01,0.001), used the CE loss with SGD with a scheduler. The scheduler helped in increasing the accuracy which was getting stalled at 50% and the model started overfitting. I also changed the scheduler from Step to ReduceLROnPlateau, the later seemed to work better. Then, changed the model, based on some suggestions from the TA, to remove the max pooling layer from the Resnet 34 model. I also stopped the model at different time intervals and manually changed the learning rate if the accuracy was saturating. Reducing the learning rate this way substantially improved the accuracy. The final model was a Resnet 34, without the max pool layer.

5. The final accuracy reached after ~50 epochs was 79.58%, beyond that it decreased using the current model.

6. Performing tests on the test set and saving the model and test results.

7. Reverse mapping to compensate the mapping done by ImageFolder and getting the correct labels and saving to a csv file.

8. Used the same model for the verification task.

9. Defined a custom dataset function to load the image pairs, with the normalisation.

10. Defined a function to compute the similarity score between two images, using Cosine similarity.

11. Trained it on the validation pairs, computed the AUC score and tested on the test pairs to get the cosine scores.

12. Saved the scored into a csv file.