import numpy as np 
import time

class Learner():
    def __init__(self, globalQueue, primaryNetwork, actionSpace):
        """
        - Uses shared memory to perform batch training across asynchronous games being played

        globalQueue    : Shared memory that all Player objects write to. Contains the agents memory for learning.
        primaryNetwork : Network that the learner will update.
        actionSpace    : Integer representing the number of discrete actions the agent can take 
        """

        self.globalQueue = globalQueue
        self.killed = False
        self.net = primaryNetwork
        self.numUpdated = 0
        self.actionSpace = actionSpace
        
    def run(self, saver,sess, modelSavePath):
        """
        -Runs the trainn method and writes the network weights to a file incrementally
        
        saver         : Tensorflow saver object, for writing the weights to file
        sess          : Reference to the tensorflow session
        modelSavePath : Path for the model to be saved to
        """

        while(not self.killed):
            self.train()
            self.numUpdated += 1
            if(self.numUpdated%20000 == 0):
                saver.save(sess, modelSavePath)
                print("Model weights have been saved")
        
    def train(self):
        """
        -Runs the training operation for the primary network
        """

        if(self.net.killAll):
            self._kill()

        empty = False
        state = []
        actions = []
        rewards = []
        while(not empty):
            example = self.globalQueue.get()
 
            for prevState, action, reward in zip(example['prevStates'], example['actions'],example['rewards']):
                state.append(np.array(prevState).reshape(-1,84,84,4))
                actions.append(np.eye(self.actionSpace)[np.array(action)].reshape(-1,self.actionSpace).astype(np.float32))
                rewards.append(np.array(reward).reshape(-1))
            empty = self.globalQueue.empty()
        
        if(len(rewards) != 0 ):
            states = np.array(state).reshape(-1, 84,84,4)
            actions = np.array(actions).reshape(-1,self.actionSpace)
            rewards = np.array(rewards).reshape(-1)
            self.net.train(states, rewards, actions)

                    
    def _kill(self):
        self.killed = True
