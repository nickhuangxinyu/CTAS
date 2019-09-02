import pandas as pd
import numpy as np
from Reader import *
from Dater import *
import sys

class DataLoader:
  def __init__(self):
    self.r = Reader()
    self.dr = Dater()

  def Load(self, file_name, future_ws=50, win=1):
    return self.r.Labelling(self.r.LoadOneDF(file_name), future_ws=future_ws, win=win)

  def GetXY(self, file_name, future_ws=50, win=1):
    df = self.Load(file_name, future_ws, win)
    if len(df) == 0:
      print('GetXY loading empty df!')
      return np.array([[]]), np.array([])
    y = df['label'].tolist()
    del df['label']
    del df['time_sec']
    del df['ticker']
    x = df.values.tolist()
    return x, y

  def GetListXY(self, file_list, future_ws=50, win=1):
    if len(file_list) == 0:
      print('meet empty filelist')
      return [[]], []
    for i, fl in enumerate(file_list):
      x,y = self.GetXY(fl, future_ws, win)
      if len(y) > 0:
        file_list = file_list[i+1:]
        break
    print('init x,y shape is %s and %s' %(np.shape(x), np.shape(y)))
    for fl in file_list:
      print('Getting XY from %s' % (fl))
      temp_x, temp_y = self.GetXY(fl, future_ws, win)
      if len(temp_x) > 0 and len(temp_y) > 0:
        print('this x,y shape is %s and %s' %(np.shape(temp_x), np.shape(temp_y)))
        x = np.append(x, temp_x, axis=0)
        y = np.append(y, temp_y, axis=0)
    return x, y

  def GetTrainTest(self, con, train_start, train_end, test_start, test_end):
    if train_end < train_start or test_start < train_end or test_end < test_start:
      print('GetTrainTest Failed! %s %s %s %s not accending' % (train_start, train_end, test_start, test_end))
      sys.exit(1)
    train_date = self.dr.GetDateList(train_start, train_end)
    test_date = self.dr.GetDateList(test_start, test_end)
    train_list = []
    test_list = []
    for td in train_date:
      file_path = '/running/'+td+'/'+con+'8888.csv'
      if os.path.exists(file_path):
        train_list.append(file_path)
    for td in test_date:
      file_path = '/running/'+td+'/'+con+'8888.csv'
      if os.path.exists(file_path):
        test_list.append(file_path)
    print('find %d trainfile %d testfile' %(len(train_list), len(test_list)))
    return train_list, test_list

if __name__ == '__main__':
  file_path = '/running/2019-05-10/ni8888.csv'
  file_list = [
    '/running/2019-05-10/ni8888.csv',
    '/running/2019-05-09/ni8888.csv',
    '/running/2019-05-08/ni8888.csv',
    '/running/2019-05-07/ni8888.csv'
  ]
  dl = DataLoader()
  x,y = dl.GetXY(file_path, win=3)
  print(np.shape(x))
  print(np.shape(y))
  x,y = dl.GetListXY(file_list, win=3)
  print(np.shape(x))
  print(np.shape(y))
