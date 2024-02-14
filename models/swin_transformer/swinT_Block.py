from keras import layers

from models.swin_transformer.init import image_dimension, patch_size, num_patch_x, num_patch_y, embed_dim, num_heads, \
    window_size, num_mlp, qkv_bias, dropout_rate, shift_size
from models.swin_transformer.patch import PatchExtract, PatchEmbedding, PatchMerging
from models.swin_transformer.st import SwinTransformer


def SwinT_Block(inp):
    x = layers.RandomCrop(image_dimension, image_dimension)(inp)
    x = layers.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation='relu')(x)

    return x