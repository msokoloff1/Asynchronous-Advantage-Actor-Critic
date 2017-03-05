import numpy as np 
import time

class Learner():
    def __init__(self, globalQueue, primaryNetwork, actionSpace):
        super().__init__()
        self.globalQueue = globalQueue
        #self.setDaemon(True)
        self.killed = False
        self.net = primaryNetwork
        self.numUpdated = 0
        self.actionSpace = actionSpace
        
        
    def run(self, saver,sess, MODELSAVEPATH):
        while(not self.killed):
            self.train()
            self.numUpdated += 1
            if(self.numUpdated%20000 == 0):
                saver.save(sess, MODELSAVEPATH)
                print("SAVED!")
        
    def train(self):
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