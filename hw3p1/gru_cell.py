import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        self.z = self.z_act(self.Wzh@h + self.Wzx@x + self.biz + self.bhz)
        self.r = self.r_act(self.Wrh@h + self.Wrx@x + self.bir + self.bhr)
        self.n = self.h_act(self.r * (self.Wnh@h + self.bhn)+ self.Wnx@x + self.bin )
        print(self.Wnh.shape)
        h_t = (1-self.z)*self.n + self.z*h

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        #raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        
        d1 = self.z * delta 
        dn_hat = (1 - self.z) * delta * self.h_act.derivative()

        
        d2 = (self.r * dn_hat)@ self.Wnh

        dz = (self.hidden - self.n) * delta
        dz_hat = dz * self.z_act.derivative()
        d3 = dz_hat@self.Wzh

        dr = dn_hat * (self.Wnh@self.hidden + self.bhn)
        dr_hat = dr * self.r_act.derivative()
        d4 = dr_hat@self.Wrh

        dh = d1 + d2 + d3 + d4
        dx_1 = np.dot(dn_hat, self.Wnx)
        dx_2 = np.dot(dz_hat,self.Wzx)
        dx_3 = np.dot(dr_hat,self.Wrx)
        dx = dx_1 + dx_2 + dx_3

        self.dWnx = np.dot(self.x.reshape(self.d,1),dn_hat).T
        self.dWnh = np.dot(self.hidden.reshape(self.h,1) , self.r * dn_hat).T
        
        self.dWrx = np.dot(self.x.reshape(self.d,1),dr_hat).T
        self.dWrh = np.dot(self.hidden.reshape(self.h,1),dr_hat).T
        self.dWzx = np.dot(self.x.reshape(self.d,1),dz_hat).T
        self.dWzh = np.dot(self.hidden.reshape(self.h,1),dz_hat).T


        self.dbin = dn_hat
        self.dbiz = dz_hat
        self.dbir = dr_hat

        self.dbhz = dz_hat
        self.dbhn = self.r * dn_hat
        self.dbhr = dr_hat
        
        #print(dh)
        #print(dx)
        
        
        
        

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        #raise NotImplementedError
