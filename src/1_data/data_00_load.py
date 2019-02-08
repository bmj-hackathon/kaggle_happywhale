# %% {"_uuid": "2cea35de3530cc898be5b85063b84e875401d092"}
os.listdir("../input/")

# %% {"_uuid": "46a8839e13a14eb8d16ea6823de9927ea63d5001"}
train_df = pd.read_csv("../input/train.csv")
train_df.head()


# %% {"_uuid": "f46b24dbba74f22833cac6140e60348b15a8e047"}
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0

    for fig in data['Image']:
        # load images into images of size 100x100x3
        img = image.load_img("../input/" + dataset + "/" + fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
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
