"""
Configuration file to import shared packages and modules in all TCoClust modules.
Modules in TCoClust should include 'from config import *', or similar import statement, if general packages or modules
are to be used.
"""


import time
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Dict, Any, Iterable, TYPE_CHECKING
from importlib.util import find_spec
from scipy.special import factorial, gammaln, logsumexp, comb
from scipy.stats.contingency import crosstab
from scipy.stats import poisson
from itertools import product
