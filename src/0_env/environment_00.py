#%%
import os

import warnings
import gc
warnings.simplefilter("ignore", category=DeprecationWarning)

from pathlib import Path
#%%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow


# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
import sklearn as sk
import sklearn.preprocessing

#%%
import keras as ks
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

#%%
