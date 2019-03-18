# %% {"_uuid": "f46b24dbba74f22833cac6140e60348b15a8e047"}

# data=train_df
# m = train_df.shape[0]
# dataset = "train"

def prepareImages(df_data, num_images, folder_path):
    print("Preparing images")
    X_train = np.zeros((num_images, 100, 100, 3))
    count = 0

    for fig in df_data['Image']:
        # load images into images of size 100x100x3

        img = tf.keras.preprocessing.image.load_img( folder_path / fig, target_size=(100, 100, 3))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return X_train


# %%
def prepare_labels(y):
    values = np.array(y)
    label_encoder = sk.preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = sk.preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    return y, label_encoder

#%%
PATH_ASSETS = Path.cwd() / 'assets'
assert PATH_ASSETS.exists()
PATH_DATASET = PATH_ASSETS / 'dataset.h5'

#%%
if PATH_DATASET.exists():
    # Load the dataset
    with h5py.File(PATH_DATASET, 'r') as hf:
        print([k for k in hf.keys()])
        # x_tr = hf['X_tr'].value
        X_tr = hf['X_tr'][()]
        X_cv = hf['X_cv'][()]
        y_tr = hf['y_tr'][()]
        y_cv = hf['y_cv'][()]

else:
    # Create the dataset and save

    X_tr = prepareImages(df_tr, df_tr.shape[0], PATH_INPUT / "train")
    X_tr /= 255

    X_cv = prepareImages(df_cv, df_cv.shape[0], PATH_INPUT / "train")
    X_cv /= 255

    # %%
    y_tr, label_encoder_tr = prepare_labels(df_tr['Id'])
    y_cv, label_encoder_cv = prepare_labels(df_cv['Id'])

    with h5py.File(PATH_DATASET, 'w') as hf:
        hf.create_dataset('X_tr', data=X_tr)
        hf.create_dataset('X_cv', data=X_cv)
        hf.create_dataset('y_tr', data=y_tr)
        hf.create_dataset('y_cv', data=y_cv)




