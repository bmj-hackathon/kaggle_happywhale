
# %%
history = model.fit(X_tr, y_tr, epochs=5, batch_size=100, verbose=1)
gc.collect()

# %%
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# %%
y_cv_pred = model.predict(X_cv)

# %%
PATH_TEST = PATH_INPUT / 'test'
test = [f.name for f in PATH_TEST.iterdir()]
print(len(test))

# %%
col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''

# %%
X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255

# %%
predictions = model.predict(np.array(X), verbose=1)

# %%
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

# %%
test_df.head(10)
test_df.to_csv('submission.csv', index=False)
