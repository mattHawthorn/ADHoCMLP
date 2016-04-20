#coding: utf-8
from .functions import *


######################################################
## ADHoCNN:                                         ##
## Almost-Done Homebrew Categorical Neural Network  ##
## Or maybe Always-Developing...                    ##
######################################################


#TODO: handle categorical y, softmax output function, cross-entropy loss function,
# optional normalization of y to help with the case of multiple regression,
# optional PCA/whitening for X, optional dropout probability, 
# optional autoencoder pretraining
# specify separation at an earlier hidden layer as opposed to only the output layer,
# to_pickle() method and CategoricalNN_from_pickle() function.
class CategoricalNN:
    """
    Neural network implementing two methods of handling categorical input.
    Present implementation assumes continuous output.
    Specify hidden layer sizes as a list.  Input and output layer sizes are determined upon training,
    and depend on the data and the method of handling categorical variables.
    activation is passed as a function directly, and currently can be any one of:
        nn.logistic, nn.arctan, nn.tanh, nn.exp, nn.identity
    loss is also given as a function and must be one of:
        nn.squared_loss
    The derivative function is determined automatically using deriv().
    bias is boolean and specifies whether to use a bias term at each of the hidden layers and the output layer.
    categorical_method must be in {'separation','one-hot'}.  'separation' specifies the method described in:
    Brouwer, Roelof K. "A feed-forward network for input that is both categorical and quantitative." Neural Networks 15.7 (2002): 881-890.
    """

    def __init__(self,h_layer_sizes,activation,loss,output_activation=identity,bias=True):
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self._d_act = deriv(activation)
        self._d_out = deriv(output_activation)
        self._d_loss = deriv(loss)
        
        if not hasattr(h_layer_sizes,'__len__'):
            h_layer_sizes = [h_layer_sizes]
        h_layer_sizes = list(h_layer_sizes)
        h_layer_sizes = [int(i) for i in h_layer_sizes]
        self.h_layer_sizes = h_layer_sizes
        self.input_layer_size = None
        self.output_layer_size = None
        
        self.weights = [None]*(len(h_layer_sizes)+1)
        if bias:
            self.biases = [None]*(len(h_layer_sizes)+1)
        else:
            self.biases = None
        
        self.initialized = False
        self.trained = False
        
    
    def train(self,X,y,learning_rate,epochs,**kwargs):
        """
        X and y are training data.  Learning 
        legal keyword args are:
        batch_size: number of training examples per mini-batch.  Defaults to X.shape[0]
        
        categorical_indices: numerical (column) indices where categorical variables are to be found
        
        categorical_method: one of {'separation','one-hot',None}.  Defaults to None if no categorical_incices
            are supplied, otherwise 'one-hot'
        
        report_interval: show progress (loss on whole data set) every report_interval epochs
        
        stochastic: boolean. Shuffle the training data on each epoch?  No effect if batch_size is default.
        
        init_weight_sd: standard deviation of initial random weights
        
        reg_lambda: optional regularization term
        
        momentum: optional momentum term
        
        normalize_learning_rate: boolean. Should the learning rate be interpreted as the rate for a single instance?
            if so, the supplied learning rate is divided by the batch size. Default is True.
            
        stopping_criterion: must be in {'relative_test','absolute_test'}. Stop training if 
            (current test error - min test error)/(min test error) > stopping_threshold, or
            current test error - min test error > stopping_threshold, respectively.
            
        test_interval: how often to compute errors on the test set, in epochs
        """
        if self.trained:
            print("NN has already been trained, and input and output layer sizes are fixed; "+
                  "call CategoricalNN.update() to continue training, or instantiate a new instance.")
            return
        
        categorical_method = kwargs.get('categorical_method',None)
        categorical_indices = kwargs.get('categorical_indices',None)
        warn_sample_size = kwargs.get('warn_sample_size',100)
        self.init_weight_sd = kwargs.get('init_weight_sd',1.0)
        self.reg_lambda = kwargs.get('reg_lambda',None)
        
        # if y is a vector, make it an nx1 matrix
        if len(y.shape) == 1:
            y = y.reshape((y.shape[0],1))
        
        #number of outputs
        self.outputs = y.shape[1]
        
        # handle input and output layer sizes appropriately according to the presence or absence of categoricals
        # and the specified way of handling them
        if categorical_indices:
            if not categorical_method:
                categorical_method = 'one-hot'
            self.categorical_method = categorical_method
            self.categorical_indices = categorical_indices
            self.quant_indices = [i for i in range(X.shape[1]) if i not in self.categorical_indices]
                    
            # all possibile values for the categorical variables
            self.categorical_values = list(map(lambda i: unique(X[:,i]), self.categorical_indices))
            # the number of unique values for each
            self.unique_levels = list(map(len, self.categorical_values))
            
            if self.categorical_method == 'separation':
                # if any output node would have fewer than this many samples to train on, issue a warning
                warn_sample_size = 100
                # output size is the product of all the categorical level counts
                self.cat_combinations = reduce(lambda x,y: x*y, self.unique_levels, 1)
                
                # TODO: actually count the sample sizes rather than use the pigeon-hole priciple
                if X.shape[0]/self.cat_combinations < warn_sample_size:
                    print("Warning: at least one combination of categoricals will yield a sample size" +
                          "smaller than {} for the {} method.".format(warn_sample_size,self.categorical_method))
                
                self.output_layer_size = self.cat_combinations*y.shape[1]
                # only the quantitative variables are inputs
                self.input_layer_size = len(self.quant_indices)
                
                # generate the output masks in advance for the different categorical combinations
                # 0-1 vectors with 0's everywhere but the index corresponding to the data's
                # categorical value combination
                self._output_mask = {}
                
                # loop over all combinations of categorical values and enumerate them with and index i
                for i, comb in enumerate(product(*self.categorical_values)):
                    # make a vector mask with length equal to the number of categorical combinations
                    mask = zeros(self.cat_combinations)
                    # unmask the entry corresponding to this combination
                    mask[i] = 1.0
                    # allow for multiple y outputs by repeating values-
                    # this results in a vector with a block of 1's where the target output should be
                    mask = repeat(mask,self.outputs)
                    # store for use in training
                    self._output_mask[comb] = mask
                    
                # self.encoder is a reserved slot for one-hot encoding, so we don't need it here
                self.encoder = None
                # set the proper input encode method. This one keeps only the quantitatve variables and 
                # normalizes them as usual
                self.encode = self.encode_separation
                
            elif self.categorical_method == 'one-hot':
                # warn if there are more than sqrt(number of samples) inputs; just a heuristic for now
                warn_inputs = sqrt(X.shape[0])
                # the only output is for y, again possibly multi-valued
                self.output_layer_size = y.shape[1]
                # the inputs are all the quant variables, plus one-hot encoded categoricals
                self.input_layer_size = len(self.quant_indices) + reduce(lambda x,y: x+y, self.unique_levels, 0)
                
                # warn if the number of parameters in the first layer has grown very large in relation to the sample size due to the encoding
                if self.input_layer_size > warn_inputs:
                    print("Warning: input layer size is {} for a training set of only {} instances".format(self.input_layer_size,X.shape[0]))
                
                # fit a one-hot encoder            
                encoder = OneHotEncoder(sparse=False)
                encoder.fit(X[:,self.categorical_indices])
                # save it for future predictions
                self.encoder = encoder
                self.encode = self.encode_one_hot
                
            else:
                raise ValueError("Unsupported categorical method: {}".format(self.categorical_method))
                
        else:
            # This is the simple purely-quantitative case
            # Output units map directly to y values
            self.output_layer_size = y.shape[1]
            # Inputs map directly to input units
            self.input_layer_size = X.shape[1]
            # All inputs are quantitative
            self.quant_indices = list(range(X.shape[1]))
            # No categoricals to deal with
            self.categorical_method = None
            self.categorical_indices = None
            # encoding is simple normalization
            self.encode = self.normalize
        
        # store the means and sd's of the quantitative columns for normalizing later test data
        self.X_mean = X[:,self.quant_indices].mean(axis=0)
        # I use degree of freedom offset = 1 here to get the unbiased normal estimator; not really of issue for large sample sizes
        self.X_sd = X[:,self.quant_indices].std(axis=0,ddof=1)
        # initialize random weights with the specified standard deviation (default is 1)
        self.init_weights(self.init_weight_sd)
        self.initialized = True
        
        # where all the work happens
        self.update(X,y,learning_rate,epochs,**kwargs)
        

    def update(self,X,y,learning_rate,epochs,**kwargs):
        """
        Shares some keyword args as train(), but this skips initialization of weights and continues training in
        a semi-online fashion, so init_weight_sd is excluded.
        Also categorical_method and categorical_indices are assumed fixed at this point, so they are not passed.
        No warning is issued for small sample sizes on the outputs in the case of the separation method.
        kwargs accepted therefore are:
            batch_size, report_interval, stochastic (bool), normalize_learning_rate,
            X_test, y_test, test_interval, stopping_criterion, stopping_threshold
        """
        if not self.initialized:
            print("NN has not been initialized with weights; call CategoricalNN.train() for initial training.")
            
        batch_size = kwargs.get('batch_size',X.shape[0])
        report_interval = kwargs.get('report_interval',None)
        verbose = kwargs.get('verbose',True)
        stochastic = kwargs.get('stochastic',True)
        stochastic = False if stochastic != True else True
        normalize_learning_rate = kwargs.get('normalize_learning_rate',True)
        momentum = kwargs.get('momentum',None)
        
        # optionally report errors on a held-out test set
        X_test = kwargs.get('X_test',None)
        y_test = kwargs.get('y_test',None)
        stopping_criterion = kwargs.get('stopping_criterion',None)
        stopping_threshold = kwargs.get('stopping_threshold',None)
        test_interval = kwargs.get('test_interval',None)
        
        if not (X_test is None and y_test is None and stopping_criterion is None and stopping_threshold is None):
            if not (X_test is not None and y_test is not None and bool(stopping_criterion)==bool(stopping_threshold)):
                raise ValueError("Invalid testing/stopping configuration. If stopping_criterion is specified," + 
                                 " stopping_threshold must be given. If either is specified, X_train and y_train" + 
                                 " must be as well.  X_train and y_train must both be present or absent.")
        
        # if y is a vector, make it an nx1 matrix; likewise for y_test above
        if len(y.shape) == 1:
            y = self._vec_to_col(y)
         
        # if using the separation method, duplicate the y matrix to match the shape of the output matrices
        # and generate a mask array of the same shape. Mask array must be generated before encoding, since it depends
        # on the categoricals, which get thrown out upon encoding.
        if self.categorical_method == 'separation':
            y = self._expand_y(y)
            mask = self.mask(X)
        
        # encode the inputs. This keeps only columns that get treated numerically,
        # which in the case of one-hot encoding includes the 0-1 encoded variables.
        # encode() handles all cases
        X = self.encode(X)
        
        if X_test is not None and y_test is not None:
            # Perform all of the same transformations of X_test and y_test if specified
            if len(y_test.shape) == 1:
                y_test = self._vec_to_col(y_test)
            if self.categorical_method == 'separation':
                y_test = self._expand_y(y_test)
                mask_test = self.mask(X_test)
            X_test = self.encode(X_test)
            loss_type = 'test'
        else:
            # Use X and y as the test sets for reporting.  This should copy by reference, not value
            X_test = X
            y_test = y
            if self.categorical_method == 'separation':
                mask_test = mask
            loss_type = 'training'
        
        # train in batches. Int division to get the number
        num_batches = X.shape[0]//int(batch_size)
        if not stochastic:
            # iterator of lists of indices, one for each batch
            batches = kBatches(arange(0,X.shape[0],dtype='int'),num_batches)
        
        # initialize the step size; if normalization is not specified, this is simply the learning rate
        # for all epochs. Otherwise, we divide by the batch size for each batch, which could change by +/-1
        # if the number of training instances is not a multiple of the batch size. 
        step = learning_rate
        # initialize the minimum loss as infinite; report when it is surpassed
        min_loss = inf
        # initialize a list to store the losses
        losses = []
        # in the case of momentum, initialize zero prior gradients
        if momentum:
            prior_W_grads = [zeros(W.shape,dtype='float') for W in self.weights]
            prior_b_grads = [zeros(b.shape,dtype='float') for b in self.biases]
        
        for epoch in range(1,epochs+1):
            # randomize training order on each epoch if specified
            if stochastic:
                batches = kBatches(permutation(X.shape[0]),num_batches)
                
            for batch in batches:
                # subset the training data from the original input
                X_train = X[batch,:]
                y_train = y[batch,:]
                
                # predictions and cached values at each hidden layer
                y_hat, output_cache = self._forward(X_train,cache_outputs=True)
                
                delta = self._d_loss(y_train,y_hat)
                
                # mask the output errors for the separation method
                if self.categorical_method=='separation':
                    delta = delta*mask[batch,:]
                    
                # pass the error gradient backward
                # returned variables are lists of arrays in the shapes of the hidden and output
                # layer weights and biases.  These are the deltas to be subtracted.
                W_grads, b_grads = self._backward(X_train,delta,y_hat,output_cache)
                
                # Should the learning rate be interpreted as the rate for a single instance?
                # if so, the supplied learning rate is divided by the batch size.
                if normalize_learning_rate:
                    step = learning_rate/X.shape[0]
        
                # add regularization term to the gradient if lambda is specified
                if self.reg_lambda:
                    for i in range(len(self.weights)):
                        W_grads[i] += self.reg_lambda*self.weights[i]
                    
                # subtract the gradients from each layer's weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= step*W_grads[i]
                    if momentum:
                        self.weights[i] -= (step*momentum)*prior_W_grads[i]
                        prior_W_grads[i] = W_grads[i].copy()
                    
                if self.biases:
                    for i in range(len(self.weights)):
                        self.biases[i] -= step*b_grads[i]
                    if momentum:
                        self.biases[i] -= (step*momentum)*prior_b_grads[i]
                        prior_b_grads[i] = b_grads[i].copy()
            
            # report progress at the specified interval, or compute test error for stopping criterion
            if (report_interval and epoch%report_interval==0) or (stopping_criterion and epoch%test_interval == 0):
                # X_test is already encoded, so we need not use the predict() method
                y_hat = self._forward(X_test, cache_outputs=False)
        
                if self.categorical_method == 'separation':
                    # mask is stored; we don't need to compute it again
                    y_hat = y_hat*mask_test
                    y_hat = self._reduce_separated(y_hat)
                    # only take one copy of the copied y matrix for computing loss
                    loss = self.loss(y_test[:,arange(0,self.outputs)],y_hat)
                else:
                    loss = self.loss(y_test,y_hat)
                
                loss = loss.mean(axis = 0)[0]
                
                if report_interval and epoch%report_interval==0:
                    if verbose:
                        print("{} loss ({}) after training epoch {}: {}.".format(self.loss.__name__, 
                                                         "Test" if stopping_criterion else "Training", epoch, loss))
                
                if stopping_criterion and epoch%test_interval == 0:
                    # compute test error and optionally stop training
                    if stopping_criterion == 'relative_test':
                        rel_loss = (loss - min_loss)/min_loss
                    elif stopping_criterion == 'absolute_test':
                        rel_loss = loss - min_loss
                    if rel_loss > stopping_threshold:
                        print("Min test loss: {}. Current test loss: {}. Relative delta: {}. Stopping.".format(min_loss,loss,stopping_threshold))
                        # Exit training
                        break
                    elif rel_loss < 0:
                        min_loss = loss
                        if verbose:
                            print("New minimum test loss ({}): {}".format(self.loss.__name__,min_loss))
                        
                else:
                    if loss < min_loss:
                        min_loss = loss
                        if verbose:
                            print("*** New minimum {} loss: {}".format(loss_type,min_loss))
                        
                # store the loss for this reporting round
                losses.append(loss)
        
        self.trained = True
        return losses
        
        
    def init_weights(self,sd):
        # initialize random weights for all layers
        # Get the dimensions of the layers:
        dims = [self.input_layer_size] + self.h_layer_sizes + [self.output_layer_size]
        
        # loop through them, generating random vars with the desired sd,
        # normalized by the input size
        for i in range(1,len(dims)):
            random_weights = sd*randn(dims[i-1], dims[i])/sqrt(dims[i-1])
            self.weights[i-1] = random_weights
            if self.biases:
                self.biases[i-1] = zeros((1,dims[i]),dtype='float')
        
    
    def encode_one_hot(self,X):
        # encoding for the one-hot method; normalize the quantitative vars, one-hot-encode
        # the categoricals, and concatenate the results
        quant = X[:,self.quant_indices]
        cat = X[:,self.categorical_indices]
        return concatenate((self.normalize(quant),self.encoder.transform(cat)), axis=1)
        
        
    def encode_separation(self,X):
        # encoding for the separation method; keep only the quantitative vars for inputs
        quant = X[:,self.quant_indices]
        return self.normalize(quant)
        
        
    def normalize(self,X):
        return (X-self.X_mean)/self.X_sd
        
    
    def mask(self,X):
        # for the separation method, return a mask for the whole X array
        mask = apply_along_axis(self._get_mask,1,X)
        return mask
    
    
    def _get_mask(self,x):
        # single-row masking method
        return self._output_mask[tuple(x[self.categorical_indices])]
        
    
    def _reduce_separated(self,y_hat):
        # assumes that y_hat has already been masked.  Achieves the equivalent of row-by-row
        # subsetting on the appropriate outputs via reshaping and summing (where masked entries
        # are zero and therefore do not contribute to the sum)
        return y_hat.reshape(y_hat.shape[0],self.cat_combinations,self.outputs).sum(axis=1)
    
    
    def _expand_y(self,y):
        return concatenate([y]*self.cat_combinations,axis=1)
    
    
    def _vec_to_col(self,y):
        return y.reshape((y.shape[0],1))
    
    
    def predict(self,X):
        # encode the inputs
        inputs = self.encode(X)
        # propagate them forward
        y_hat = self._forward(inputs, cache_outputs=False)
        
        if self.categorical_method == 'separation':
            y_hat = y_hat*self.mask(X)
            y_hat = self._reduce_separated(y_hat)
        
        return y_hat
        
        
    def _forward(self,X,cache_outputs=False):
        # pass encoded inputs X forward, keeping the outputs of the hidden layers
        # if specified, for use in back-propagation of errors
        if cache_outputs:
            cache = []
        
        # make a copy to prevent corruption of the data
        out = X.copy()
        
        # forward propagation in the case of bias terms
        if self.biases:
            for i in range(len(self.weights)-1):
                out = self.activation(dot(out,self.weights[i])+self.biases[i])
                if cache_outputs:
                    cache.append(out)
            # last output stage is the prediction
            y_hat = self.output_activation(dot(out,self.weights[-1])+self.biases[-1])
        # forward propagation in the case of no bias terms
        else:
            for i in range(len(self.weights)-1):
                out = self.activation(dot(out,self.weights[i]))
                if cache_outputs:
                    cache.append(out)
            # last output stage is the prediction
            y_hat = self.output_activation(dot(out,self.weights[-1]))
        
        if cache_outputs:
            return y_hat, cache
        else:
            return y_hat
        
        
    def _backward(self,X,delta,y_hat,layer_outputs):
        # initialize lists to hold gradients
        W_grads = [None]*len(self.weights)
        if self.biases:
            b_grads = [None]*len(self.weights)
        else:
            b_grads = None
        
        # rate of change of y_hat w.r.t the linear combinations at the output layer
        delta = delta*self._d_out(y_hat)
        
        # iterate over all but the first layer's weights
        # layer outputs has length 1 less than 
        for i in range(len(self.weights)-1,0,-1):
            # dot the outputs of the previous layer with the derivative of the
            # outputs of the present layer
            d_W = dot(layer_outputs[i-1].T,delta)
            W_grads[i] = d_W
            if self.biases:
                # summing the columns is equivalent to matrix muliplying by an all-1's matrix
                d_b = delta.sum(axis=0, keepdims=True)
                b_grads[i] = d_b
            
            delta = dot(delta,self.weights[i].T)*self._d_act(layer_outputs[i-1])
        
        # First layer; inputs to the weighting are simply X
        W_grads[0] = dot(X.T, delta)
        b_grads[0] = delta.sum(axis=0, keepdims=True)
        
        return W_grads, b_grads




class kBatches():    
    def __init__(self,keys,k):
        self.keys = keys
        self.k = k
        self.foldsize = float(len(keys))/k
        self._i = None
        
    def __iter__(self):
        self._i=0.0
        return self
        
    def __next__(self):
        low = int(round(self._i,10))
        if low>=len(self.keys):
            raise StopIteration
        high = int(round(self._i + self.foldsize,10))
        sample = self.keys[low:high]
        
        self._i+=self.foldsize
        return sample
    
