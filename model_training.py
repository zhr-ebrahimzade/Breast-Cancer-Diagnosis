from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam


def build_model(input_shape, num_classes):
    DenseModel = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in DenseModel.layers[:149]:
        layer.trainable = False
    for layer in DenseModel.layers[149:]:
        layer.trainable = True

    model = Sequential()
    model.add(DenseModel)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.7))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu', kernel_initializer="he_normal"))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer="he_normal"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, training_set, train_labels, validation_data, callbacks, epochs=6, batch_size=128):
    model.fit(x=training_set, y=train_labels, batch_size=batch_size, epochs=epochs,
              callbacks=callbacks, validation_data=validation_data)
    model.save('/content/drive/MyDrive/bestModel/BestModel.h5')
