from keras import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Input, GlobalAveragePooling2D, Dense

from models.swin_transformer.init import num_classes


def basic_block(inputs, filters, strides=(1, 1)):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    if strides != (1, 1) or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet18(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = basic_block(x, filters=64)
    x = basic_block(x, filters=64)
    x = basic_block(x, filters=128, strides=(2, 2))
    x = basic_block(x, filters=128)
    x = basic_block(x, filters=256, strides=(2, 2))
    x = basic_block(x, filters=256)
    x = basic_block(x, filters=512, strides=(2, 2))
    x = basic_block(x, filters=512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, output)
    return model