import gym
import numpy as np
from scipy.misc import imresize
import queue
import copy
from gym import wrappers


class Game():
    def __init__(self, gameType, imageDims, numFrames, metrics = False):
        """
        This class adds an abstraction on top of dimensionality reduction for the open ai environment
        """
        if(metrics):
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
        assert int(state.shape[-1]) == 3, "Dimensions incorrect for generating grayscale"
        return ( (np.mean(state,-1)/128)-1.0).reshape(84,84,1)
        
        

    def _updateState(self, state):
        #something is wrong, prev state should be one behind curstate..
        if(self.once):
            self.prevState.get()
            self.prevState.put(self.prevHolder)

        self.currentState.get()  # <= dequeue
        self.currentState.put(state)  # <=Enqueue
        self.prevHolder = state
        self.once = True

    def step(self, move):
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
        self.once = False
        origState = self.env.reset()
        resize = lambda x: imresize(x, (self.imageDims, self.imageDims), 'bilinear').reshape(84,84,3)
        self.currentState = queue.Queue()
        [self.currentState.put(self._colorToBW(resize(origState))) for _ in range(self.numFrames)]
        self.prevState = queue.Queue()
        [self.prevState.put(self._colorToBW(resize(origState))) for _ in range(self.numFrames)]

    def _getPrevState(self):
        return self._queueToNumpy(self.prevState)

    def _queueToNumpy(self, queueObj):
        frames = []
        for _ in range(self.numFrames):
            state = queueObj.get()
            frames.append(state)
            queueObj.put(state)

        a = np.concatenate(frames, axis = 2)
        return a
