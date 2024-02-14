num_classes = 4
input_shape = (128, 128, 3)
patch_size = (2, 2)
dropout_rate = 0.03
num_heads = 8
embed_dim = 64
num_mlp = 256
qkv_bias = True
window_size = 2
shift_size = 1
image_dimension = 128

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 32
num_epochs = 100
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1