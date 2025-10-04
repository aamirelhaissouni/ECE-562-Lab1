from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ##initialize weights and biases
        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b2 = np.zeros(num_classes) 

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ## get parameters
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        ## first layer affine - relu
        hidden, cache1 = affine_relu_forward(X, W1, b1)

        ## second layer affine
        scores, cache2 = affine_forward(hidden, W2, b2)


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ## compute loss and gradient
        loss, dscores = softmax_loss(scores, y)

        ## add regularization to loss
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        ## backward pass throughs second layer
        dhidden, dW2, db2 = affine_backward(dscores, cache2)

        ##backwards pass through first layer
        dX, dW1, db1 = affine_relu_backward(dhidden, cache1)

        ## add regularization to gradients
        dW1 += self.reg * W1
        dW2 += self.reg * W2

        #store gradients
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layer_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            self.params['W' + str(i+1)] = np.random.normal(0, weight_scale, (layer_dims[i], layer_dims[i+1]))
            self.params['b' + str(i+1)] = np.zeros(layer_dims[i+1])

            if self.normalization in ['batchnorm', 'layernorm'] and i < self.num_layers - 1:
                self.params['gamma' + str(i+1)] = np.ones(layer_dims[i+1])
                self.params['beta' + str(i+1)] = np.zeros(layer_dims[i+1])

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Dictionary to store caches for backward pass
        cache = {}
        current_input = X

        # Forward pass through hidden layers (with ReLU activation)
        for i in range(self.num_layers - 1):
            W = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]
            
            # Check if normalization is used
            if self.normalization == 'batchnorm':
                # Affine forward
                affine_out, affine_cache = affine_forward(current_input, W, b)
                
                # Batch normalization
                gamma = self.params['gamma' + str(i+1)]
                beta = self.params['beta' + str(i+1)]
                bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, self.bn_params[i])
                
                # ReLU
                relu_out, relu_cache = relu_forward(bn_out)
                
                # Store caches
                cache['affine' + str(i+1)] = affine_cache
                cache['bn' + str(i+1)] = bn_cache
                cache['relu' + str(i+1)] = relu_cache
                
                current_input = relu_out
                
            elif self.normalization == 'layernorm':
                # Affine forward
                affine_out, affine_cache = affine_forward(current_input, W, b)
                
                # Layer normalization
                gamma = self.params['gamma' + str(i+1)]
                beta = self.params['beta' + str(i+1)]
                ln_out, ln_cache = layernorm_forward(affine_out, gamma, beta, self.bn_params[i])
                
                # ReLU
                relu_out, relu_cache = relu_forward(ln_out)
                
                # Store caches
                cache['affine' + str(i+1)] = affine_cache
                cache['ln' + str(i+1)] = ln_cache
                cache['relu' + str(i+1)] = relu_cache
                
                current_input = relu_out
                
            else:
                # No normalization - use convenience function
                current_input, cache[i+1] = affine_relu_forward(current_input, W, b)
            
            # Dropout (if enabled)
            if self.use_dropout:
                current_input, dropout_cache = dropout_forward(current_input, self.dropout_param)
                cache['dropout' + str(i+1)] = dropout_cache

        # Final output layer (no ReLU, no normalization, no dropout)
        W_final = self.params['W' + str(self.num_layers)]
        b_final = self.params['b' + str(self.num_layers)]
        scores, final_cache = affine_forward(current_input, W_final, b_final)
        cache['final'] = final_cache


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute softmax loss and gradient
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization to loss only for Ws not bs
        for i in range(self.num_layers):
            W = self.params['W' + str(i+1)]
            loss += 0.5 * self.reg * np.sum(W * W)

        # Backward pass through final layer
        dout = dscores
        dx_final, dW_final, db_final = affine_backward(dout, cache['final'])
        grads['W' + str(self.num_layers)] = dW_final + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db_final

        dout = dx_final

        # Backward pass through hidden layers (in reverse order)
        for i in range(self.num_layers - 1, 0, -1):
            # Dropout backward (if used)
            if self.use_dropout:
                dout = dropout_backward(dout, cache['dropout' + str(i)])
            
            # Backward through layer based on normalization type
            if self.normalization == 'batchnorm':
                # ReLU backward
                drelu = relu_backward(dout, cache['relu' + str(i)])
                # Batchnorm backward
                dbn, dgamma, dbeta = batchnorm_backward(drelu, cache['bn' + str(i)])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
                # Affine backward
                dout, dW, db = affine_backward(dbn, cache['affine' + str(i)])
                
            elif self.normalization == 'layernorm':
                # ReLU backward
                drelu = relu_backward(dout, cache['relu' + str(i)])
                # Layernorm backward
                dln, dgamma, dbeta = layernorm_backward(drelu, cache['ln' + str(i)])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
                # Affine backward
                dout, dW, db = affine_backward(dln, cache['affine' + str(i)])
                
            else:
                # No normalization - use convenience function
                dout, dW, db = affine_relu_backward(dout, cache[i])
            
            # Store gradients with L2 regularization (only for weights)
            grads['W' + str(i)] = dW + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
