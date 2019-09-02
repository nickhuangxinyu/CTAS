from datetime import datetime
import pandas as pd
class Dater:
  def __init__(self):
    pass
  def GetDateList(self, beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

if __name__ == '__main__':
  d=Dater()
  print(d.datelist('20190101', '20190301'))
