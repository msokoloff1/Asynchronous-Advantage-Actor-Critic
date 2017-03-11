import gym
import numpy as np
from scipy.misc import imresize
import queue
import copy
from gym import wrappers


class Game():
    def __init__(self, gameType, imageDims, numFrames, monitor = False):
        """
        - This class adds an abstraction on top of the OpenAI gym
        
        gameType  : String containing the name of the OpenAI environment to be built
        imageDims : The width and height for rescaling the observation
        numFrames : Number of frames to stack in order to capture time dependencies 
        monitor   : Boolean indicating whether or not to use the OpenAI Monitor wrapper
        """

        if(monitor):
            self.env = wrappers.Monitor(gym.make(gameType), 'experiments/one')
        else:
            self.env = gym.make(gameType)
            
        self.numActions = self.env.action_space.n
        self.imageDims = imageDims
        self.numFrames = numFrames
        self.obsSpace = (imageDims, imageDims, numFrames)
        self.currentState = queue.Queue()
        self.reset()
        self.prevHolder = None
        self.once = False
        
    def _colorToBW(self, state):
        """
        -Converts and scales an observation of the agent's state 
 
        state : Three channel numpy array

        Returns a one channel numpy array containing the grayscaled state
        """
        assert int(state.shape[-1]) == 3, "Dimensions incorrect for generating grayscale"
        return ( (np.mean(state,-1)/128)-1.0).reshape(84,84,1)
        
        

    def _updateState(self, state):
        """
        - Updates the state of the agent after taking an action
 
        state : One channel numpy array containing the new state 
        """ 

        if(self.once):
            self.prevState.get()
            self.prevState.put(self.prevHolder)

        self.currentState.get()  # <= dequeue
        self.currentState.put(state)  # <=Enqueue
        self.prevHolder = state
        self.once = True

    def step(self, move):
        """
        - Abstration on top of OpenAI's step method. Keeps track of multiple timesteps, resizes, and grayscales image.

        move : The index corresponding to the move the agent has selected
        """

        state, reward, terminal, info = self.env.step(move)
        state = imresize(state, (self.imageDims, self.imageDims), 'bilinear').reshape(84,84,3)
        bwState = self._colorToBW(state)
        self._updateState(bwState)
        return [self._queueToNumpy(self.currentState)
            , self._getPrevState()
            , reward
            , terminal
            , info
                ]

    def reset(self):
        """
        -Abstration on top of OpenAI's reset method. Resets the environment.
        """

        self.once = False
        origState = self.env.reset()
        resize = lambda x: imresize(x, (self.imageDims, self.imageDims), 'bilinear').reshape(84,84,3)
        self.currentState = queue.Queue()
        [self.currentState.put(self._colorToBW(resize(origState))) for _ in range(self.numFrames)]
        self.prevState = queue.Queue()
        [self.prevState.put(self._colorToBW(resize(origState))) for _ in range(self.numFrames)]

    def _getPrevState(self):
        """
        returns the previous state as a numpy array
        """

        return self._queueToNumpy(self.prevState)

    def _queueToNumpy(self, queueObj):
        """
        - Turns a queue object into a numpy array
        
        queueObj : The queue to be turned into a numpy array
        
        Returns a numpy array that contains all of the states in the queue       
        """

        frames = []
        for _ in range(self.numFrames):
            state = queueObj.get()
            frames.append(state)
            queueObj.put(state)

        a = np.concatenate(frames, axis = 2)
        return a
