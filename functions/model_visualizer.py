from keras.utils import plot_model

def modelVisualize(model):
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)