# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        batch_size = x.shape[0]
        #in_channel = x.shape[1]
        input_size = x.shape[-1]
        output_size = (input_size-self.kernel_size)//self.stride + 1
        
        z = np.zeros([batch_size,self.out_channel,output_size])
        for i in range(batch_size):
            for j in range(self.out_channel):
                for m in range(output_size):
                    start = m*self.stride
                    end = m*self.stride + self.kernel_size
                    segment = self.x[i,:,start:end]
                    z[i,j,m] = np.sum(self.W[j,:,:]*segment)
                    z[i,j,m] = z[i,j,m] + self.b[j]
        return z                       
                    
                    
                    
                    

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size = delta.shape[0]
        out_channel = delta.shape[1]
        input_size = self.x.shape[-1]
        output_size = delta.shape[-1]
        self.dx = np.zeros([batch_size, self.in_channel, input_size])
        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(output_size):
                    self.db[j] = self.db[j] + delta[i,j,k]
                    
        for i in range(batch_size):
                for j in range(output_size):
                    for k in range(out_channel):
                        for l in range(self.in_channel):
                            for m in range(self.kernel_size):
                                segment = self.x[i,l,m+self.stride*j]
                                self.dW[k,l,m] = self.dW[k,l,m] + segment*delta[i,k,j]
                                self.dx[i,l,m+self.stride*j] = self.dx[i,l,m+self.stride*j] + self.W[k,l,m]*delta[i,k,j]
                                
                                
        return self.dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size = x.shape[0]
        in_channel = x.shape[1]
        input_width = x.shape[-2]
        input_height = x.shape[-1]
        output_width = (input_width-self.kernel_size)//self.stride + 1
        output_height = (input_height-self.kernel_size)//self.stride + 1
        out_channel = self.out_channel
        z = np.zeros([batch_size,out_channel,output_width, output_height])
        for i in range(batch_size):
            for j in range(out_channel):
                for m in range(output_width):
                    for n in range(output_height):
                        width_start = m*self.stride
                        width_end = m*self.stride + self.kernel_size
                        height_start = n*self.stride
                        height_end = n*self.stride + self.kernel_size
                        segment = x[i,:,width_start:width_end,height_start:height_end]
                        z[i,j,m,n] = np.sum(self.W[j,:,:,:]*segment)
                        z[i,j,m,n] = z[i,j,m,n] + self.b[j]
                    
        return z      
        
        
        
        
        

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size = delta.shape[0]
        input_width = self.x.shape[-2]
        input_height = self.x.shape[-1]
        output_width = delta.shape[-2]
        output_height = delta.shape[-1]
        out_channel = delta.shape[1]
        self.dx = np.zeros([batch_size, self.in_channel, input_width, input_height])
        for i in range(batch_size):
            for j in range(out_channel):
                for m in range(output_width):
                    for n in range(output_height):
                        self.db[j] = self.db[j] + delta[i,j,m,n]
                    
                    
        for i in range(batch_size):
                for w in range(output_width):
                    for h in range(output_height):
                        for c in range(out_channel):
                            for l in range(self.in_channel):
                                for m in range(self.kernel_size):
                                    for n in range(self.kernel_size):
                                        segment = self.x[i,l,m+self.stride*w,n+self.stride*h]
                                        self.dW[c,l,m,n] = self.dW[c,l,m,n] + segment*delta[i,c,w,h]
                                        self.dx[i,l,m+self.stride*w,n+self.stride*h] = self.dx[i,l,m+self.stride*w,n+self.stride*h] + self.W[c,l,m,n]*delta[i,c,w,h]
                                
        return self.dx

        


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (self.kernel_size-1)*(self.dilation-1) + self.kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        # TODO: padding x with self.padding parameter (HINT: use np.pad())
       
        batch_size = x.shape[0]
        in_channel = x.shape[1]
        input_width = x.shape[-2]
        input_height = x.shape[-1]
        x = np.pad(x, ((0,0),(0,0),(self.padding,self.padding), (self.padding,self.padding)), 'constant', constant_values = (0,0))
        self.x = x
        input_width_padding = input_width + 2*self.padding
        input_height_padding = input_height + 2*self.padding
        
        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated
        for i in range(self.out_channel):
            for j in range(self.in_channel):
                W_ini_m=0
                for m in range(0,self.W_dilated.shape[-2],self.dilation):
                    W_ini_n=0
                    for n in range(0,self.W_dilated.shape[-1],self.dilation):
                        self.W_dilated[i,j,m,n] = self.W[i,j,W_ini_m,W_ini_n]
                        W_ini_n=W_ini_n + 1
                    W_ini_m = W_ini_m+1
                        
                        
        # TODO: regular forward, just like Conv2d().forward()
        output_width = (input_width_padding-self.kernel_dilated)//self.stride + 1
        output_height = (input_height_padding-self.kernel_dilated)//self.stride + 1
        out_channel = self.out_channel
        z = np.zeros([batch_size,out_channel,output_width, output_height])
        
        for i in range(batch_size):            
            for j in range(out_channel):
                for m in range(output_width):
                    for n in range(output_height):
                        width_start = m*self.stride
                        width_end = m*self.stride + self.kernel_dilated
                        height_start = n*self.stride
                        height_end = n*self.stride + self.kernel_dilated
                        segment = x[i,:,width_start:width_end,height_start:height_end]
                        z[i,j,m,n] = np.sum(self.W_dilated[j,:,:,:]*segment)
                        z[i,j,m,n] = z[i,j,m,n] + self.b[j]
                    
        return z      
        #raise NotImplementedError


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.

        batch_size = delta.shape[0]
        input_width = self.x.shape[-2]
        input_height = self.x.shape[-1]
        output_width = delta.shape[-2]
        output_height = delta.shape[-1]
        out_channel = delta.shape[1]
        self.dx = np.zeros([batch_size, self.in_channel, self.x.shape[-2], self.x.shape[-1]])
        self.dW_dilated = np.zeros(self.W_dilated.shape)
        for i in range(batch_size):
            for j in range(out_channel):
                for m in range(output_width):
                    for n in range(output_height):
                        self.db[j] = self.db[j] + delta[i,j,m,n]
                      
                        
        
        for i in range(batch_size):
                for w in range(output_width):
                    for h in range(output_height):
                        for c in range(out_channel):
                            for l in range(self.in_channel):
                                for m in range(self.kernel_dilated):
                                    for n in range(self.kernel_dilated):
                                        segment = self.x[i,l,m+self.stride*w,n+self.stride*h]
                                        self.dW_dilated[c,l,m,n] = self.dW_dilated[c,l,m,n] + segment*delta[i,c,w,h]
                                        self.dx[i,l,m+self.stride*w,n+self.stride*h] = self.dx[i,l,m+self.stride*w,n+self.stride*h] + self.W_dilated[c,l,m,n]*delta[i,c,w,h]
        for i in range(self.out_channel):
            for j in range(self.in_channel):
                W_ini_m=0
                for m in range(0,self.dW_dilated.shape[-2],self.dilation):
                    W_ini_n=0
                    for n in range(0,self.dW_dilated.shape[-1],self.dilation):
                        self.dW[i,j,W_ini_m,W_ini_n] = self.dW_dilated[i,j,m,n]
                        W_ini_n=W_ini_n + 1
                    W_ini_m = W_ini_m+1  
        self.dx = self.dx[:,:,self.padding:-self.padding,self.padding:-self.padding]
        return self.dx
        #raise NotImplementedError



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        x = x.reshape(self.b, self.c*self.w)
        return x

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        batch_size = delta.shape[0]
        in_channel = self.c
        in_width = self.w
        return delta.reshape(batch_size,in_channel,in_width)