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
    buy_count = (pred_y==1).sum()
    sell_count = (pred_y==2).sum()
    buy_success = (y[np.where(pred_y == 1)] == 1).mean()*100
    buy_fail = (y[np.where(pred_y == 1)] == 2).mean()*100
    sell_success = (y[np.where(pred_y == 2)] == 2).mean()*100
    sell_fail = (y[np.where(pred_y == 2)] == 1).mean()*100
    print('buy:count %d, success %.1f%%, fail %.1f%%, sell:count %d, success %.1f%%, fail %.1f%%' %(
    buy_count, buy_success, buy_fail, sell_count, sell_success, sell_fail))
    return buy_count, buy_success, buy_fail, sell_count, sell_success, sell_fail

if __name__ == '__main__':
  er = Evaluator()
  er.Acc([0,0,1,2,0,1], [1,1,2,2,0,1])
