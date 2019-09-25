import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import sys
sys.path.append('../preprocessing')
sys.path.append('../evaluator')
from DataLoader import *
from evaluator import *
import time

train_list = [ 
  #'/running/2019-05-07/ni8888.csv',
  #'/running/2019-05-08/ni8888.csv',
  #'/running/2019-05-10/ni8888.csv',
  '/root/crypto_cache/crypto2019-08-01.log'
]

test_list = [ 
  '/root/crypto_cache/crypto2019-08-09.log'
  #'/running/2019-05-09/ni8888.csv'
  #'/running/2019-04-16/ni8888.csv'
  ]

def main():
  dl = DataLoader()
  er = Evaluator()
  train_list, test_list = dl.GetTrainTest('ni', '2019-01-01', '2019-01-05', '2019-06-08', '2019-06-10')
  #train_list, test_list = dl.GetTrainTest('BTCUSDT', '2019-08-01', '2019-08-03', '2019-08-07', '2019-08-09')
  print(train_list)
  win = 1
  train_x, train_y = dl.GetListXY(train_list)
  train_x = np.reshape(train_x, (len(train_x), -1))
  train_x = preprocessing.scale(train_x)
  print('train x, y shape is %s %s' %(np.shape(train_x), np.shape(train_y)))
  clf = SVC(gamma='auto')
  print('fitting svm model')
  start_sec = time.time()
  clf.fit(train_x, train_y)
  print('finished fitting svm model, cost %f s' %(time.time()-start_sec))
  test_x, test_y = dl.GetListXY(test_list, win = win)
  test_x = np.squeeze(test_x, axis=2)
  test_x = preprocessing.scale(test_x)
  pred_y = clf.predict(test_x)
  er.Acc(test_y, pred_y)
  '''
  pred_y = np.array(pred_y)
  test_y = np.array(test_y)
  print('test_y pred_y shape %s %s' % (np.shape(test_y), np.shape(pred_y)))
  print('pred_y buy %.1f%% sell %.1f%% buy' %((pred_y==1).mean()*100, (pred_y==2).mean()*100))
  print('buy acc %f, sell acc %f' %((test_y[np.where(pred_y == 1)] == 1).mean(), (test_y[np.where(pred_y == 2)] == 2).mean()))
  #print(clf.score(test_x, test_y))
  '''

if __name__ == '__main__':
  main()
