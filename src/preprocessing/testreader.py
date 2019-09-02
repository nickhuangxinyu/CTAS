import numpy as np
import pandas as pd
from Reader import *

r=Reader()
a = r.LoadOneDF('/running/2019-02-18/ni8888.csv')
r.Labelling(a)
