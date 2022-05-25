import numpy as np
from ctc import *



class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------
        
        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            #print(logits.shape)
            target = self.target[b][:target_lengths[b]]
            logits = self.logits[:input_lengths[b],b,:]
           
            extSymbols, skipConnect = CTC.targetWithBlank(self,target)
            alpha = CTC.forwardProb(self,logits,extSymbols,skipConnect)
            beta = CTC.backwardProb(self,logits,extSymbols,skipConnect)
            gamma = CTC.postProb(self,alpha,beta)
           
            [T,S] = gamma.shape
            #print(S,logits.shape)
            for t in range(T):
                for i in range(S):
                    
                    totalLoss[b] -= gamma[t,i]*np.log(logits[t,extSymbols[i]])
            
            
            # -------------------------------------------->

            # Your Code goes here
            #raise NotImplementedError
            # <---------------------------------------------
        
        
        totalLoss=np.sum(totalLoss,axis=0)/B
        return totalLoss

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            target = self.target[b][:self.target_lengths[b]]
            logits = self.logits[:self.input_lengths[b],b,:]
           
            extSymbols, skipConnect = CTC.targetWithBlank(self,target)
            alpha = CTC.forwardProb(self,logits,extSymbols,skipConnect)
            beta = CTC.backwardProb(self,logits,extSymbols,skipConnect)
            gamma = CTC.postProb(self,alpha,beta)
           
            [T,S] = gamma.shape
            #print(S,logits.shape)
            for t in range(T):
                for i in range(S):
                    dY[t,b,extSymbols[i]] -= gamma[t,i]/logits[t,extSymbols[i]]
            

            # Your Code goes here
            #raise NotImplementedError
            # <---------------------------------------------

        return dY
