# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"_uuid": "88e20188ddffce29fc4c3af7b6fc09bb48cc1fee"}
# ## Training 

# %% [markdown] {"_uuid": "132b11ca41afe41627ed3c0df8b2be39d30f93d2"}
# Let's start by importing our libararies.

# %% {"_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5", "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19"}
from fastai.conv_learner import *
from fastai.dataset import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import math

# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
MODEL_PATH = 'Resnet18_v1'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
SAMPLE_SUB = '../input/sample_submission.csv'


# %% [markdown] {"_uuid": "e0624ab350e370dbff80cac45f33744c48e5633b"}
# The architecture is flexible, I chose Resnet18 since it can fit quite well into a kernel. You may play with this if you want to. 

# %% {"_uuid": "6ea9033e0200d3d9142b4ee05c45c1dd4f2d8c1d"}
arch = resnet18
nw = 4

# %% [markdown] {"_uuid": "bf2a5c5342e855974efaeb7fe5c2b90f2cf636cf"}
# Next, we prapare out dataset to work with Fastai's pipeline.

# %% {"_uuid": "d9adfc15b56c7f80f291c66dc6d6f38d4d55e6a2"}
train_df = pd.read_csv(LABELS).set_index('Image')
unique_labels = np.unique(train_df.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
print("Number of classes: {}".format(len(unique_labels)))
train_names = train_df.index.values
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)
test_names = [f for f in os.listdir(TEST)]

# %% [markdown] {"_uuid": "6a910097d19053c50d60ea7ee9496ed2a55746e2"}
# Let's draw a simple histogram to see the sample-per-class distribution.

# %% {"_uuid": "ddef1744553be7723709a1e14253612a18c6f7e2"}
labels_count = train_df.Id.value_counts()
_, _,_ = plt.hist(labels_count,bins=100)
labels_count

# %% [markdown] {"_uuid": "0a3a2f26fd6613685c994bf0a4514a276aa5f047"}
# Ugh, okay, let's kick the elephant out of the room and try again

# %% {"_uuid": "fcd012819740e8f7c2ebcc0274c8547f90434629"}
print("Count for class new_whale: {}".format(labels_count[0]))

plt.hist(labels_count[1:],bins=100,range=[0,100])
plt.hist(labels_count[1:],bins=100,range=[0,100])

# %% [markdown] {"_uuid": "53cd78eae54b541399f324a8531ffd1fbca90a7d"}
# So most of the classes have only one or two sample(s), making **train_test_split** directly on the data impossible. We'll try a simple fix by duplicating the minor classes so that each class have a minimum of 5 samples.

# %% {"_uuid": "9374cfc59c2cedef5ca6b4292e1c4e350c31e638"}


# %% {"_uuid": "5c9a8d0b818a13929e3b18771165956b3f751d7d"}
dup = []
for idx,row in train_df.iterrows():
    if labels_count[row['Id']] < 5:
        dup.extend([idx]*math.ceil((5 - labels_count[row['Id']])/labels_count[row['Id']]))
train_names = np.concatenate([train_names, dup])
train_names = train_names[np.random.RandomState(seed=42).permutation(train_names.shape[0])]
len(train_names)

# %% {"_uuid": "0fc648f57bcc32e48bb043da3854eb46f2d91540"}
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42069)
for train_idx, val_idx in sss.split(train_names, np.zeros(train_names.shape)):
    tr_n, val_n = train_names[train_idx], train_names[val_idx]
print(len(tr_n), len(val_n))

# %% [markdown] {"_uuid": "037cdcbea35cf6a0785e5707202b26b5f707b716"}
# The image sizes seem to vary, so we'll try to see what the average width and height are:

# %% {"_uuid": "7062c686741f15788a24f6f1aecc0b5d9ce57e8e"}
avg_width = 0
avg_height = 0
for fn in os.listdir(TRAIN)[:1000]:
    img = cv2.imread(os.path.join(TRAIN,fn))
    avg_width += img.shape[1]
    avg_height += img.shape[0]
avg_width //= 1000
avg_height //= 1000
print(avg_width, avg_height)

# %% [markdown] {"_uuid": "a855425946822696c5aa4d37401f3b5c1d0a88c5"}
# They turn out to be quite big, especially the width, so below you'll see I resize everything back to **average_width/4**. You may consider continue training on bigger size, but that probably won't fit in a kernel. 

# %% {"_uuid": "94af91d70819db979d39a4d77b2e30493498978b"}
class HWIDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        # We crop the center of the original image for faster training time
        img = cv2.resize(img, (self.sz, self.sz))
        return img

    def get_y(self, i):
        if (self.path == TEST): return 0
        return self.train_df.loc[self.fnames[i]]['Id']


    def get_c(self):
        return len(unique_labels)


# %% [markdown] {"_uuid": "267ee406354b98daa4682d7fcb6f08106bd7bee6"}
#

# %% {"_uuid": "140dfae2b41cbe4f770f8d80fcaba0ebc772e983"}
class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b, self.c = b, c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  # add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1 / (c - 1) if c < 0 else c + 1
        x = lighting(x, b, c)
        return x
    
def get_data(sz, bs):
    aug_tfms = [RandomRotateZoom(deg=20, zoom=2, stretch=1),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO),
                RandomBlur(blur_strengths=3,tfm_y=TfmType.NO),
                RandomFlip(tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(HWIDataset, (tr_n[:-(len(tr_n) % bs)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, TEST))
    md = ImageData("./", ds, bs, num_workers=nw, classes=None)
    return md


# %% {"_uuid": "f8258255beb8fb608abb8a292b07c7161580007e"}
# sz = (avg_width//2, avg_height//2)
batch_size = 64
md = get_data(avg_width//4, batch_size)
learn = ConvLearner.pretrained(arch, md) 
learn.opt_fn = optim.Adam

# %% [markdown] {"_uuid": "c2a3accea73d13e6b8f81febd988b9e377fb5572"}
# Uncomment these lines to run Fastai's automatic learning rate finder. 
#

# %% {"_uuid": "04b0332bd91ee3752b8da857d34e566c96a638d4"}
# learn.lr_find()
# learn.sched.plot()
lr = 5e-3

# %% [markdown] {"_uuid": "115459f2b3756d4f029cb223a57a80abba5f2992"}
# We start by training only the newly initialized weights, then unfreeze the model and finetune the pretrained weights with reduced learning rate.

# %% {"_uuid": "f3118d5e2dbe61c8d51d0e33642ea5bb0b516a54"}
learn.fit(lr, 1, cycle_len=2)
learn.unfreeze()
lrs = np.array([lr/10, lr/20, lr/40])
learn.fit(lrs, 4, cycle_len=4, use_clr=(20, 16))
learn.fit(lrs/4, 2, cycle_len=4, use_clr=(10, 16))
learn.fit(lrs/16, 1, cycle_len=4, use_clr=(10, 16))

# %% [markdown] {"_uuid": "7b3867b2a2b604c2dfb91b94b49a96980d5883ec"}
# May be keep training on bigger image for potential performance boost.

# %% {"_uuid": "fcb801182b1440e0c29b4626510a49d93ac91d6e"}
# batch_size = 32
# md = get_data(avg_width//2, batch_size)
# learn.set_data(md)
# learn.fit(lrs/4, 3, cycle_len=2, use_clr=(10, 8))


# %% {"_uuid": "d66865dfd58fe3eac08450d646040c2550900675"}
# batch_size = 16
# md = get_data(avg_width, batch_size)
# learn.set_data(md)
# learn.fit(lrs/16, 1, cycle_len=4, use_clr=(10, 8))

# %% [markdown] {"_uuid": "eabe555ace20e3b36c6266432108d31de1e58282"}
# ## Predictions

# %% {"_uuid": "6cbfaedbad6bac01b06d87eaf3723dd260b7a51e"}
# preds_t,y_t = learn.predict_with_targs(is_test=True) # Predicting without TTA
preds_t,y_t = learn.TTA(is_test=True,n_aug=8)
preds_t = np.stack(preds_t, axis=-1)
preds_t = np.exp(preds_t)
preds_t = preds_t.mean(axis=-1)

# %% [markdown] {"_uuid": "82b4e79d2a05267790200de4860fd25a0a669f7f"}
# Finally, our submission.

# %% {"_uuid": "f5fbd91e970d375debc3270ebd5b08bb41eeb66e"}
sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.Image)
pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(learn.data.test_ds.fnames,pred_list))
pred_list_cor = [' '.join(pred_dic[id]) for id in sample_list]
df = pd.DataFrame({'Image':sample_list,'Id': pred_list_cor})
df.to_csv('submission.csv'.format(MODEL_PATH), header=True, index=False)
df.head()
