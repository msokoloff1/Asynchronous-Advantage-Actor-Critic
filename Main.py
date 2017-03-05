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
args = parser.parse_args()

gameType = 'PongDeterministic-v0'# "Breakout-v3" #'PongDeterministic-v0' #
e = game.make(gameType)
actionSpace = e.action_space.n
log = Logger(tf, "tensorBoard/testm")
learningRate = 0.0003
decay = 0.99
momentum = 0.0
epsilon = 0.1
imageDims = 84
numFrames = 4
gamma = 0.99

if(args.train):
    numLearners   = 1
    numPredictors = 1
    numPlayers = 12
    
    MODELSAVEPATH = "savedModels/modelPongm.ckpt"
    
    sess = tf.Session()
    trainingQueue = Queue(maxsize = 250)
    
    players = [Player(gameType
                      , log
                      , imageDims
                      , numFrames
                      , trainingQueue
                      ,gamma
                      , id) for id in range(numPlayers)] 
    
    PrimaryNet = Network(actionSpace
                        , log
                        , learningRate
                        , decay
                        , momentum
                        , epsilon
                        ,sess
                        ,imageDims
                        , numFrames)
    
    
    predictors = [Predictor(players, PrimaryNet) for _ in range(numPredictors)]
    saver = tf.train.Saver()
    try:
        saver.restore(sess, MODELSAVEPATH)
        print("Successfully Restored Model!!")
    except:
        sess.run(tf.global_variables_initializer())
        print("No model available for restoration")
    

    l = Learner(trainingQueue, PrimaryNet, actionSpace)    
    for player in players:
        player.start()
        
    for predictor in predictors:
        predictor.start()
        
    l.run(saver, sess, MODELSAVEPATH)

if(args.test):
    with tf.Session() as sess:
        PrimaryNet = Network(actionSpace
                        , log
                        , learningRate
                        , decay
                        , momentum
                        , epsilon
                        ,sess
                        ,imageDims
                        , numFrames)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, MODELSAVEPATH)
            print("Successfully Restored Model!!")
        except:
            print("No model found.. exiting")
            exit(0)
            
        player = Demonstrator(PrimaryNet, gameType, 84, 4)
        player.play(10)

    
    
    
    