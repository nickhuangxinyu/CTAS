import pandas as pd
import numpy as np
from Reader import *
from Dater import *
from algo import *
import sys
import random
import time

class DataLoader:
  def __init__(self, future_ws = 50, win = 1):
    self.future_ws = future_ws
    self.win = win
    self.r = Reader()
    self.dr = Dater()

  def Load(self, file_name):
    return self.Labelling(self.r.LoadOneDF(file_name))
    #return self.r.LoadOneDF(file_name)

  def Labelling(self, df, cut_tail = False): 
    if len(df) == 0:
      print('labelling passed empty df in!')
      return df
    df = df[df['asks[0]'] - df['bids[0]'] > 0]
    min_move = (df['asks[0]'] - df['bids[0]']).min()
    self.label_map = {'Flat': 0, 'Buy': 1, 'Sell': 2}
    if 'label' in df.columns:
      print('df has labelled!')
      return df
    ask_min = list(map(lambda x: -x, GetSlipWindowMax((df['asks[0]']*-1).tolist(), self.future_ws)))
    bid_max = GetSlipWindowMax(df['bids[0]'].tolist(), self.future_ws)
    df = df.drop([i for i in range(len(df)-self.future_ws, len(df))])
    ask_min = ask_min[self.future_ws:]
    bid_max = bid_max[self.future_ws:]
    #print('len(df)=%d, len(ask_min)=%d' % (len(df), len(bid_max)))
    df.insert(len(df.columns), 'label', 0)
    df.loc[df['bids[0]']-ask_min >= self.win*min_move, 'label'] = self.label_map['Sell']
    df.loc[bid_max-df['asks[0]'] >= self.win*min_move, 'label'] = self.label_map['Buy']
    if cut_tail:
      df.loc[[i for i in range(500)], 'label'] = self.label_map['Flat']
      df.loc[[i for i in range(len(df)-500, len(df))], 'label'] = self.label_map['Flat']
    #print('after labelling: %.1f%% are buy, %.1f%% are sell, %.1f%% are flat' %((df['label']==self.label_map['Buy']).mean()*100, (df['label']==self.label_map['Sell']).mean()*100, (df['label']==self.label_map['Flat']).mean()*100))
    return df

  def GetXY(self, file_name, cut_tail = True):
    start_time = time.time()
    df = self.Load(file_name)
    if len(df) == 0:
      print('GetXY loading empty df!')
      return np.array([[]]), np.array([])
    y = df['label'].tolist()
    del df['label']
    del df['time_sec']
    del df['ticker']
    start_time = time.time()
    x = rolled(df, self.future_ws, cut_tail)
    if cut_tail:
      y = y[self.future_ws:]
    print('GetXY cost %.2fs' %(time.time()-start_time))
    return x, y

  def GetListXY(self, file_list):
    if len(file_list) == 0:
      print('meet empty filelist')
      return [[]], []
    print('GetListXY started: %s-%s' % (file_list[0], file_list[-1]))
    start_time = time.time()
    for i, fl in enumerate(file_list):
      x,y = self.GetXY(fl)
      if len(y) > 0:
        file_list = file_list[i+1:]
        break
    x, y = np.array(x), np.array(y)
    #print('init x,y shape is %s and %s' %(np.shape(x), np.shape(y)))
    for fl in file_list:
      #print('Getting XY from %s' % (fl))
      temp_x, temp_y = self.GetXY(fl)
      if len(temp_x) > 0 and len(temp_y) > 0:
        #x = np.append(x, temp_x, axis=0)
        x = np.vstack((x, temp_x))
        y = np.hstack((y, temp_y))
        #y = np.append(y, temp_y, axis=0)
    print('GetListXY finished: %.1f%% buy, %.1f%% sell, cost %.1fs' % ((y==1).mean()*100, (y==2).mean()*100, time.time()-start_time))
    return x, y

  def GetTrainTest(self, con, train_start, train_end, test_start, test_end, add_post=True):
    if train_end < train_start or test_end < test_start:
      print('GetTrainTest Failed! %s %s %s %s not accending' % (train_start, train_end, test_start, test_end))
      sys.exit(1)
    train_date = self.dr.GetDateList(train_start, train_end)
    test_date = self.dr.GetDateList(test_start, test_end)
    train_list = []
    test_list = []
    for td in train_date:
      file_path = '/running/'+td+'/'+con+ ("8888" if add_post else '') + '.csv'
      print(file_path)
      if os.path.exists(file_path):
        train_list.append(file_path)
    for td in test_date:
      file_path = '/running/'+td+'/'+con+ ('8888' if add_post else '') + '.csv'
      if os.path.exists(file_path):
        test_list.append(file_path)
    print('find %d trainfile %d testfile' %(len(train_list), len(test_list)))
    return train_list, test_list

  def GetBatch(self, x, y, batch_size = 128):
    start_index = np.random.randint(len(x) - self.future_ws, size = batch_size)
    rx, ry = [], []
    for si in start_index:
      #rx.append(x[si:si+self.future_ws])
      #ry.append(self.OneHot(y[si+self.future_ws-1]).tolist()[0])
      rx.append(x[si])
      #ry.append(self.OneHot(y[si]).tolist()[0])
      ry.append(self.LabelToClass(y[si]))
    return rx, ry

  def LabelToClass(self, y):
    r=[]
    if isinstance(y, list):
      for l in y:
        if l == self.label_map['Flat']:
          r.append([1, 0, 0])
        elif l == self.label_map['Buy']:
          r.append([0.1, 0.9, 0])
        else:
          r.append([0.1, 0, 0.9])
      return r
    if y == self.label_map['Flat']:
      return [1, 0, 0]
    elif y == self.label_map['Buy']:
      return [0.1, 0.9, 0]
    else:
      return [0.1, 0, 0.9]

  def OneHot(self, y, C=3):
      return np.eye(C)[np.array(y).reshape(-1)]

if __name__ == '__main__':
  file_path = '/running/2019-05-10/ni8888.csv'
  file_list = [
    '/running/2019-05-10/ni8888.csv',
    '/running/2019-05-09/ni8888.csv',
    '/running/2019-05-08/ni8888.csv',
    '/running/2019-05-07/ni8888.csv'
  ]
  dl = DataLoader()
  x,y = dl.GetXY(file_path)
  print(np.shape(x))
  print(np.shape(y))
  x,y = dl.GetListXY(file_list)
  print(np.shape(x))
  print(np.shape(y))
