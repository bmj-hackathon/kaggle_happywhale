# %% [markdown] {"_uuid": "da035fe58e548e8b1b7e8e89725b9e6bc745aa7b"}
# # Humpback Whale Identification - CNN with Keras
# This kernel is based on [Anezka Kolaceke](https://www.kaggle.com/anezka)'s awesome work: [CNN with Keras for Humpback Whale ID](https://www.kaggle.com/anezka/cnn-with-keras-for-humpback-whale-id)

import os
from pathlib import Path
# os.environ["PATH"] = "/usr/local/cuda-9.0/bin" + os.pathsep + os.environ["PATH"]
# os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64"
assert "LD_LIBRARY_PATH" in os.environ
assert "/usr/local/cuda-9.0/bin" in [p for p in os.environ['PATH'].split(':')]
# for e in os.environ['PATH'].split(':'):
#     print(e)


# %% {"_uuid": "0d9c73ad23e6c2eae3028255ee00c3254fe66401"}
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

import keras

from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
# import
# dir(tf.keras.applications)
from keras.applications.imagenet_utils import preprocess_input
# tf.keras.applications.imagenet_utils

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


print("tensorflow", tf.VERSION)
print("keras", tf.keras.__version__)

# %% {"_uuid": "2cea35de3530cc898be5b85063b84e875401d092"}
INPUT_DIR = Path('./input')
assert INPUT_DIR.exists()
os.listdir(INPUT_DIR)

# %% {"_uuid": "46a8839e13a14eb8d16ea6823de9927ea63d5001"}
train_df = pd.read_csv(INPUT_DIR/ "train.csv")
train_df.head()

# %% {"_uuid": "f46b24dbba74f22833cac6140e60348b15a8e047"}
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        this_path = INPUT_DIR / dataset / fig
        img = image.load_img(this_path, target_size=(100, 100, 3))
        # img = image.load_img(INPUT_DIR+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train

# %% {"_uuid": "6587a101b58af064af0f9c60a1070c6c8f52d45f"}
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

# %% {"_uuid": "4afe4128a0cd6859848c8a80686208082d647c39"}
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255

# %% {"_uuid": "675924f8863aef27cf90dc668e0a68cd609dfc1c"}
y, label_encoder = prepare_labels(train_df['Id'])

# %% {"_uuid": "14d243b19023e830b636bea16679e13bc40deae6"}
y.shape

# %% {"_uuid": "e7af799d186a1b97b6aa325d7d576a1fb55a6c5d"}
model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

# %% {"_uuid": "169f45e150c3a584e0f655a8eda523e0675da63a"}
history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()

# %% {"_uuid": "7bca48a1d0963cbf70685b75431435cef9499895"}
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# %% {"_uuid": "debe961c93b72bef151d9aad3ca2cb500ee00aaa"}
test = os.listdir("../input/test/")
print(len(test))

# %% {"_uuid": "72ed8198f519f7b1ae3efbc688933c78d8cdd0e4"}
col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''

# %% {"_uuid": "52262195fc0b8755cff78bf8c98e6116d50f79af"}
X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255

# %% {"_uuid": "88c8d8ff98fbdb1df4218abb6bd51889e855a6fb"}
predictions = model.predict(np.array(X), verbose=1)

# %% {"_uuid": "66f0bdde31b8c7847916268aa82d9a1bdc9c0658"}
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

# %% {"_uuid": "09d7c1eb9b554e4e580b0c3c7eb609c15636892d"}
test_df.head(10)
test_df.to_csv('submission.csv', index=False)
