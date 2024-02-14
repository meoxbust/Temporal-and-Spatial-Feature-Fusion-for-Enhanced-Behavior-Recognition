import keras
import tensorflow_addons as tfa

def modelDefine(inputs=None, outputs=None, pretrain=False, model=None):
    Model = None
    if pretrain == False:
        Model = keras.Model(inputs=inputs, outputs=outputs)
    else:
        Model = model
    Model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    return Model