#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging

#Delete Jupyter notebook root logger handler
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)-3s - %(module)-10s  %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(levelno)-3s - %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(funcName)-10s: %(message)s"
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
#DATE_FMT = "%H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

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
#%%

def mm2inch(value):
    return value/25.4
PAPER = {
    "A3_LANDSCAPE" : (mm2inch(420),mm2inch(297)),
    "A4_LANDSCAPE" : (mm2inch(297),mm2inch(210)),
    "A5_LANDSCAPE" : (mm2inch(210),mm2inch(148)),
}
