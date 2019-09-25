import tensorflow as tf
import numpy as np
import pandas as pd
import sys 
sys.path.append('../preprocessing')
sys.path.append('../evaluator')
from DataLoader import *
from evaluator import *
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

shot = MarketSnapshot()

class lstm:
  def __init__(self, lstm_units, future_ws = 50):
    self.lstm_units = lstm_units
    self.future_ws = future_ws
    self.build_model()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.loss_hist = []

  '''
  def softmax(self, x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)
  '''

  def weighted_softmax_cross_entropy(self, y_pred, y, weight=[1.0,2.0,3.0]):
    #inty = np.array([int(i) for i in y], dtype=int32)
    y_pred = tf.nn.softmax(y_pred)
    weight = np.array(weight)/np.sum(weight)
    return -tf.reduce_sum(y*tf.log(y_pred)*weight)

  def build_model(self):
    self.x = tf.placeholder(tf.float32, [None, self.future_ws, 27])
    self.y = tf.placeholder(tf.float32, [None, 3])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_units, forget_bias=1.0)
    hiddens, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.x, dtype=tf.float32)
    nn = [50]
    self.out = hiddens[:,-1,:]
    current_layer_size = self.lstm_units
    for n in nn:
      w1 = tf.Variable(tf.random_normal([current_layer_size, n]))
      b1 = tf.Variable(tf.random_normal([1, n]))
      self.out = tf.nn.relu(tf.matmul(self.out, w1) + b1)
      current_layer_size = n
    w = tf.Variable(tf.random_normal([nn[-1], 3]))
    b = tf.Variable(tf.random_normal([1, 3]))
    self.out = tf.matmul(self.out, w) + b
    #print(self.out.shape)
    #sys.exit(1)
    #tf.nn.cros
    #tf.nn.tanh(self.out)
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.y))
    #self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=tf.argmax(self.y, 1)))
    #self.loss = self.weighted_softmax_cross_entropy(self.out, self.y, [10.0, 1.0, 1.0])
    self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

  def fit(self, x, y):
    _, loss = self.sess.run([self.train_op, self.loss], feed_dict = {self.x:x, self.y:y})
    self.loss_hist.append(loss)

  def pred(self, x):
    r = self.sess.run(self.out, feed_dict={self.x: x})
    return [np.argmax(one_hot) for one_hot in r]

  def plot_loss(self):
    plt.plot(self.loss_hist)
    plt.show()

def scale(x, future_ws):
  return np.reshape(preprocessing.scale(np.reshape(x, (len(x), -1))), (len(x), future_ws, -1)) if len(x) > 1 else x
  #return np.reshape(MinMaxScaler().fit_transform(np.reshape(x, (len(x), -1))), (len(x), future_ws, -1)) if len(x) > 1 else x

def main():
  future_ws = 10
  epochs = 2000
  show_count = 100
  win = 200
  dl = DataLoader(future_ws=future_ws, win = win)
  er = Evaluator()
  #train_list, test_list = dl.GetTrainTest('cu', '2019-01-01', '2019-02-01', '2019-01-01', '2019-02-01')
  train_list, test_list = dl.GetTrainTest('BTCUSDT', '2019-07-03', '2019-07-05', '2019-08-01', '2019-08-02', add_post = False)
  train_x, train_y = dl.GetListXY(train_list)
  test_x, test_y = dl.GetListXY(test_list)
  test_x = test_x[:int(len(test_x)/future_ws)*future_ws]
  test_y = test_y[:int(len(test_y)/future_ws)*future_ws]
  #print(np.shape(train_x), np.shape(train_y))
  #sys.exit(1)
  #train_x = reshape(preprocessing.scale(np.reshape(train_x, (len(train_x), -1))), (len(train_x, future_ws, -1))) if len(train_x) > 1 else train_x
  train_x = scale(train_x, future_ws)
  test_x = scale(test_x, future_ws)
  #train_x = preprocessing.MinMaxScaler().fit_transform(train_x)
  l = lstm(lstm_units = 100, future_ws=future_ws)
  start_time = time.time()
  train_x = train_x[:int(len(train_x)/future_ws)*future_ws]
  train_y = train_y[:int(len(train_y)/future_ws)*future_ws]
  max_iv = 0.0
  for i in range(epochs):
    bx, by = dl.GetBatch(train_x, train_y, 2048)
    tx = np.reshape(bx, (-1, future_ws, 27))
    l.fit(tx, by)
    if i % show_count == 1:
      print('%d epochs finished, cost %.1fs' % (i, time.time()-start_time))
      start_time = time.time()
      cls = l.pred(train_x[100000:200000])
      #train_real_y = np.reshape(train_y, (-1, future_ws))
      #train_real_y = [tr[-1] for tr in train_real_y]
      #print('pred and cls len is %d %d' % (len(train_real_y), len(cls)))
      print('acc is %.2f%%'%((cls==train_y[100000:200000]).mean()*100))
      bc, bs, bf, sc, ss, sf = er.Acc(train_y[100000:200000], cls)
      b_ok = bs-bf
      s_ok = ss-sf
      if b_ok + s_ok > max_iv:
        max_iv = b_ok+s_ok
        print('strat testing:')
        cls = l.pred(np.reshape(test_x, (-1, future_ws, 27)))
        er.Acc(test_y, cls)
  l.plot_loss()
  '''
  test_x, test_y = dl.GetListXY(test_list)
  test_x = scale(test_x, future_ws)
  #test_x = preprocessing.MinMaxScaler().fit_transform(test_x)
  test_x = test_x[:int(len(test_x)/future_ws)*future_ws]
  test_y = test_y[:int(len(test_y)/future_ws)*future_ws]
  cls = l.pred(np.reshape(test_x, (-1, future_ws, 27)))
  er.Acc(test_y, cls)
  '''

if __name__ == '__main__':
  main()
