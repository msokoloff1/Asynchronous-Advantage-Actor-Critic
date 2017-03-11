from Network import Network
import tensorflow as tf
from Log import Logger
from multiprocessing import Process, Queue, cpu_count
from Demonstration import Demonstrator
from Learner import Learner
from Predictor import Predictor
from Player import Player
import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-train', default = False, type=bool, help = 'Flag indicating whether to train or not')
parser.add_argument('-test', default = False, type=bool, help = 'Flag indicating whether to train or not')
parser.add_argument('-game', default = 'PongDeterminisitic-v0',type=str,  help = 'Flag indicating which game the agent should play')
parser.add_argument('-learning_rate', default=0.00025  , type=float, help="Learning rate to be used when applying gradients")
parser.add_argument('-decay', default=0.99  , type=float, help="Decay rate to be used by the RMSProp optimizer")
args = parser.parse_args()

gameType = args.game
info = gym.spec(gameType).make()
actionSpace = info.action_space.n
log = Logger(tf, "tensorBoard/"+gameType)
learningRate = args.learning_rate <= delete
decay = args.decay  <= delete
imageDims = 84
numFrames = 4
savePath = "savedModels/" + args.game + ".ckpt"


def loadWeights(path, sess, saver,train = True):
    try:
        saver.restore(sess, path)
        print("Successfully Restored Model!!"
    except:
	if(train):
            sess.run(tf.global_variables_initializer())
	    print("No model available for restoration")
        else:
	    print("No model found.. exiting")
            exit(0)
	      
	      
	      
def train(learningRate, decay,  momentum = 0.0, epsilon = 0.1, gamma = 0.99, numLearners = 1, numPredictors = 1, numPlayers = 20):
    sess = tf.Session()
    trainingQueue = Queue(maxsize = 250)
    players = [Player(gameType, log , imageDims, numFrames, trainingQueue, gamma, id) 
							  for id in range(numPlayers)] 
    PrimaryNet = Network(actionSpace, log, learningRate, decay, momentum, epsilon,sess,imageDims, numFrames) 
    predictors = [Predictor(players, PrimaryNet) for _ in range(numPredictors)]

    saver = tf.train.Saver()
    loadWeights(savePath, sess, saver,train = True)
    l = Learner(trainingQueue, PrimaryNet, actionSpace)    
    for player in players:
        player.start()
        
    for predictor in predictors:
        predictor.start()
        
    l.run(saver, sess, savePath)

def test():
    with tf.Session() as sess:
        PrimaryNet = Network(actionSpace, log,None,None,None,None,sess,imageDims,numFrames)
        saver = tf.train.Saver()
        loadWeights(savePath, sess, saver,train = False)   
        player = Demonstrator(PrimaryNet, gameType, 84, 4)
        player.play(10)

if(args.train):
    train(args.learning_rate, args.decay)

if(args.test):
    test()
    
    
