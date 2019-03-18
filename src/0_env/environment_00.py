#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging

logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

#%%
import warnings
import gc
warnings.simplefilter("ignore", category=DeprecationWarning)

from pathlib import Path


#%%
# Ensure CUDA paths!
import os
from pathlib import Path
assert "LD_LIBRARY_PATH" in os.environ
assert "/usr/local/cuda-9.0/bin" in [p for p in os.environ['PATH'].split(':')]

#%%
# Scientific stack
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import sklearn.preprocessing
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# import sklearn.preprocessing
import sklearn.model_selection
import sklearn as sk

import h5py
# %%
# Deep learning stack
# TODO: This should also be available in tensorflow, eliminate the keras dep
from keras.applications.imagenet_utils import preprocess_input
# import keras
# keras.applications.imagenet_utils.preprocess_input
import tensorflow as tf

# tf.keras.layers


# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing import image
# import
# dir(tf.keras.applications)
# from keras.applications.imagenet_utils import preprocess_input
# tf.keras.applications.imagenet_utils

# from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
# from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
# from tensorflow.keras.models import Model
#
# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Sequential

#%%

def mm2inch(value):
    return value/25.4
PAPER = {
    "A3_LANDSCAPE" : (mm2inch(420),mm2inch(297)),
    "A4_LANDSCAPE" : (mm2inch(297),mm2inch(210)),
    "A5_LANDSCAPE" : (mm2inch(210),mm2inch(148)),
}
