import tensorflow as tf
import numpy as np


class Network():
    def __init__(self,actionSpace, log, learningRate, decay, momentum, epsilon,sess,  imageDims, numFrames):
        """
        - Constructor for the global network object.

        actionSpace  : The number of possible discrete actions the agent can make
        log          : A reference to the global logging object
        learningRate : Learning rate to be used when applying gradients
        decay        : Decay parameter for the RMSProp optimizer
        momentum     : Momentum parameter for the RMSProp optimizer
        epsilon      : Epsilon parameter for the RMSProp optimizer
        sess         : A reference to the tensorflow session
        imageDims    : The width and height for rescaling the observation
        numFrames    : Number of frames to stack in order to capture time dependencies  

        """

        self.actionSpace    = actionSpace
        self.obsPH          = tf.placeholder(tf.float32, shape = [None,imageDims, imageDims, numFrames])
        self.obsRewardPH    = tf.placeholder(tf.float32, shape = [None])
        self.oheChosenActionPH = tf.placeholder(tf.float32, shape = [None, actionSpace])
        self.beta = 0.01
        self.nonZeroOffset = 0.000001
        self.log = log
        self.sess = sess
        self.curIter = 0
        self.avgRewards = tf.Variable(0.0, trainable = False)
        self.log.addSummary("expAvgReward", self.avgRewards)
        
        with tf.variable_scope("GlobalNetwork"):
            self._buildNetwork()
            self._buildLoss()
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GlobalNetwork")
        self._buildOptimizer(learningRate, decay, momentum, epsilon,  clipNorm = 80.0)
        self.log.ready(sess)
        
        self.summary_op = self.log.getSummaryOp()
        self.killAll = False
        

    def _buildNetwork(self):
        """
        - Establishes the logic for the forward pass of the network
        """

        #Core Network (reusable)
        conv1 = self._convLayer(self.obsPH, filterDims = 8,   outputFilters = 16, strides = 4,name="conv1", activation = tf.nn.relu)
        conv2 = self._convLayer(conv1     , filterDims = 4,   outputFilters = 32, strides = 2,name="conv2", activation = tf.nn.relu)
        fc1   = self._fcLayer  (conv2     , numOutputs = 256,name="fc1", activation = tf.nn.relu )

        #Outputs
        self.valueEst = tf.squeeze(self._fcLayer(fc1, numOutputs = 1,name="valueEst", activation = lambda x: x), axis=[1])
        self.policyProb  = self._fcLayer(fc1, numOutputs = self.actionSpace,name="policyProb", activation = tf.nn.softmax)
        
        #Log results
        self.log.addHist("valueEst", self.valueEst)
        self.log.addHist("policyProb", self.policyProb)
         
    def _buildLoss(self):
        """
        - Creates the loss function to be reduced during training
        """

        logXAdvantageLoss = tf.log(
                                tf.reduce_sum( 
                                    (self.policyProb * self.oheChosenActionPH) 
                                       , axis = 1)
                                       + self.nonZeroOffset) \
                           * (self.obsRewardPH - tf.stop_gradient(self.valueEst))
                         
        EntropyRegLoss  = -1 * self.beta * tf.reduce_sum( (tf.log(self.policyProb + self.nonZeroOffset)*self.policyProb) , axis = 1)
        totalPolicyLoss   =  -1 * (tf.reduce_sum(logXAdvantageLoss, axis = 0) + tf.reduce_sum(EntropyRegLoss, axis = 0))
        
        sub = (self.obsRewardPH - self.valueEst)
        valueEstLoss = 0.5 * tf.reduce_mean(tf.square(sub), axis = 0)
        
        #Final loss function to minimize
        self.totalLoss = totalPolicyLoss + valueEstLoss
        
        #Logging for tensorboard
        self.log.addHist("logXAdvantageLoss", logXAdvantageLoss)
        self.log.addHist("EntropyRegLoss", EntropyRegLoss)
        self.log.addHist("totalPolicyLoss", totalPolicyLoss)
        self.log.addHist("valueEstLoss", valueEstLoss)
        self.log.addHist("totalLoss", self.totalLoss)
        
    def _buildOptimizer(self, 
                        learningRate, decay, momentum, epsilon, clipNorm):
        """
        - Creates a graph node for applying reducing the loss (self.applyGrads)
       
        learningRate : Learning rate to be applied to gradients 
        decay        : Discount for past gradients
        momentum     : Gradient momentum
        epsilon      : non zero offset
        clipNorm     : Maximum average norm gradient allowed
        """

        optimizer = tf.train.RMSPropOptimizer(
                                          learning_rate = learningRate
                                        , decay         = decay
                                        , momentum      = momentum
                                        , epsilon       = epsilon
                                    )
        
        grads = optimizer.compute_gradients(loss)
        clippedGrads = [(tf.clip_by_average_norm(grad, clipNorm),var) for grad,var in grads]
        self.applyGrads = optimizer.apply_gradients(clippedGrads)
           
            
    def _convLayer(self, input, filterDims, outputFilters,strides,name,  activation = tf.nn.elu):
        """
        -Creates a convolutional layer
        
        input         : The previous layer that convolution will be applied to
        filterDims    : Length of the sides for a square filter
        outputFilters : Number of output feature maps
        strides       : The stride length for both the x and y dimensions
        name          : Name for this layer, used for debugging
        activation    : The nonlinearity to be applied to the result of the convolution

        Returns a tensor containing the activations for a convolutional layer
        """

        with tf.variable_scope(name):
            inputDim = int(input.get_shape()[-1])
            xavierAbs = tf.div(1.,tf.sqrt( float(inputDim) * (filterDims**2) ))
            shape     = [filterDims,filterDims, inputDim, outputFilters]
            
            weights   = tf.Variable(tf.random_uniform(
                                      shape
                                    , minval = -xavierAbs
                                    , maxval = xavierAbs
                                    ))
            
            self.log.addHist("weights" + name, weights)
            
            bias      = tf.Variable(tf.random_uniform(
                                      [outputFilters]
                                    , minval = -xavierAbs
                                    , maxval = xavierAbs
                                    ))
            
            self.log.addHist("bias" + name, bias)
            
            conv = tf.nn.conv2d(input, weights, strides = [1,strides, strides,1], padding = "SAME") 
            return activation(conv + bias)
        
    def _fcLayer(self, input, numOutputs, name, activation = tf.nn.elu):
        """
        -Creates a fully connected layer

        input       : The previous layer to be used as the input
        numOutputs  : The number of output nodes to be useds
        name        : Name for this layer, used for debugging
        activation  : The nonlinearity to be applied to the output nodes

        Returns a tensor containing the activations of the output nodes
        """

        #Flatten if the previous layer is a convolutional layer
        if(len(input.get_shape()) == 4):
                sum = np.multiply.reduce([int(x) for x in input.get_shape()[1:]])
                input = tf.reshape(input, shape = (-1,sum))
                
        inputDim = int(input.get_shape()[-1])
        xavierAbs = tf.div(1.0,tf.sqrt(float(inputDim)))
        
        shape = [inputDim, numOutputs]
        
        weights   = tf.Variable(tf.random_uniform(
                                      shape
                                    , minval = -xavierAbs
                                    , maxval = xavierAbs
                                    ))
        
        self.log.addHist("weights" + name, weights)

        bias      = tf.Variable(tf.random_uniform(
                                      [numOutputs]
                                    , minval = -xavierAbs
                                    , maxval = xavierAbs
                                    ))
        
        self.log.addHist("bias" + name, bias)
        
        fc = tf.matmul(input, weights)
        return activation(fc + bias)
        
    def gatedActivation(self, input):
	#Since this is an activation function, the input is a convolutional layer
        inputFilters = int(input.get_shape())[-1]
	conv = self._convLayer(input, 4,inputFilters*2,1,"gated",activation = lambda x: x):
        convS1, convS2 = tf.split(conv, num_or_size_splits=2, axis=-1)
        tan = tf.nn.tanh(convS1)
        sig = tf.nn.sigmoid(convS2)
        multiplied = tf.mul([tan,sig], axis =-1)
        finalConv = self._convLayer(multiplied, 1, inputFilters, "gatedOneByOne", activation = tf.nn.relu)
        residualSum = finalConv + input #NEED TO RESHAPE... getting on plane. finish later..
	return residualSum
  
    def train(self, states, rewards, actions):
        """
        -Runs the training operation
        
        states  : Numpy array containing the observed states one step before the corresponding rewards. 
                  Shape => (number of examples,  imageDims, imageDims, numFrames) 
        rewards : Numpy array containing the observed rewards proceeding each of the states
        actions : A one hot encoded numpy array containing the action chosen for each of the states
        """

        dict = {
                  self.obsPH              : states
                , self.obsRewardPH        : rewards
                , self.oheChosenActionPH  : actions
            }
        
	#Log every logFrequency number of updates        
        if(self.curIter%self.log.logFrequency == 0):
            self.sess.run(self.avgRewards.assign(self.log.avgRewards.value))
            summary, _ = self.sess.run([self.summary_op, self.applyGrads], feed_dict = dict)
            self.log.writeSummaries(summary, int(self.log.gamesPlayed.value))
        else:
            _ = self.sess.run([self.applyGrads], feed_dict = dict)

        self.curIter += 1
        

    def predict(self, states):
	"""
	-Makes a prediction for which action to take

        states : Numpy array containing the observed states.
                  Shape => (number of examples,  imageDims, imageDims, numFrames)        

        Returns a probability for each possible action.
	"""
        dict = {
              self.obsPH : states
            }
        
        policyProb,  valueEst  = self.sess.run([self.policyProb, self.valueEst], feed_dict = dict)
        
        return [policyProb,  valueEst] 
        
        
        
        
