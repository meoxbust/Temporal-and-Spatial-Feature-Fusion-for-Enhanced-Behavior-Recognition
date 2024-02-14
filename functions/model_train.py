from tensorflow.python.keras.callbacks import EarlyStopping


def modelTrain(model, inputs_tr, inputs_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        inputs_tr,
        ytr,
        batch_size=16,
        epochs=50,
        verbose=1,
        validation_data=(inputs_val, yval),
        callbacks = [early_stopping]
    )
    return history