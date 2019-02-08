
# %% {"_uuid": "7bca48a1d0963cbf70685b75431435cef9499895"}
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# %% {"_uuid": "debe961c93b72bef151d9aad3ca2cb500ee00aaa"}
PATH_TEST = PATH_INPUT / 'test'
test = [f.name for f in PATH_TEST.iterdir()]
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
