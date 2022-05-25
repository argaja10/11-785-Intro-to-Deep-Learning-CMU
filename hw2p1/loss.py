# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.softmaxce = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        self.softmaxce = np.exp(self.logits)/np.sum(np.exp(self.logits),axis=1)[:,None]
        self.loss = -np.sum(self.labels * np.log(self.softmaxce), axis=1)

        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.softmaxce - self.labels

class L2loss(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(L2loss, self).__init__()
        self.softmaxce = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        self.softmaxce = np.exp(self.logits)/np.sum(np.exp(self.logits),axis=1)[:,None]
        self.dsoftmaxce = np.diagflat(self.softmaxce.reshape(-1,1)) - np.dot(self.softmaxce, self.softmaxce.T)
        self.loss = 1 / 2 * np.sum((self.labels - self.logits)**2, axis=1)

        return self.loss
    

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return (self.softmaxce-self.labels)*self.dsoftmaxce
