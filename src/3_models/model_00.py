classification_shape = y_tr.shape[1]

# %%
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(tf.keras.layers.BatchNormalization(axis = 3, name = 'bn0'))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPooling2D((2, 2), name='max_pool'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.AveragePooling2D((3, 3), name='avg_pool'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation="relu", name='rl'))
model.add(tf.keras.layers.Dropout(0.8))
model.add(tf.keras.layers.Dense(classification_shape, activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()


