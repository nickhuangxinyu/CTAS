import numpy as np
import pandas as pd

class Evaluator:
  def __init__(self):
    pass

  def Acc(self, y, pred_y):
    y = np.array(y)
    pred_y = np.array(pred_y)
    print('pred_y buy %.1f%% sell %.1f%% buy' %((pred_y==1).mean()*100, (pred_y==2).mean()*100))
    print('y buy %.1f%% sell %.1f%% buy' %((y==1).mean()*100, (y==2).mean()*100))
    print('buy:success %f, fail %f, sell:success %f, fail %f' %((y[np.where(pred_y == 1)] == 1).mean(),
    (y[np.where(pred_y == 1)] == 2).mean(), (y[np.where(pred_y == 2)] == 2).mean(),
    (y[np.where(pred_y == 2)] == 1).mean()))
