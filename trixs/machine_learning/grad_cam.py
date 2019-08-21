import cv2
from keras.layers.core import Lambda
import tensorflow as tf
from keras.models import clone_model
import keras.backend as K
import numpy as np

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(input_model, image:np.array, category_index:int,
             kernel_size=8, image_size=100, classes=3):
    """

    :param input_model: Keras model.
    :param image: Input vector
    :param category_index: Which output classified category you want
    :param kernel_size: How large the kernel is on your data
    :param image_size: The size of your image vectors
    :param classes: Number of unique classifiers classified by the output
    :return:
    """

    # Copy model which will be augmented
    model = clone_model(input_model)
    model.set_weights(model.get_weights())

    # Add a layer which zeroes out the
    target_layer = lambda x: target_category_loss(x, category_index, classes)

    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))

    # Compute the final value of the output classifier
    loss = K.sum(model.layers[-1].output)

    # Assumes that the final convolutional layer is the first one
    conv_output = model.layers[0].output

    # Gradients of classifier wrt to the convolution layer's output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    output, grads_val = gradient_function([image])

    # Get relevant components of data
    output, grads_val = output[0, :], grads_val[0, :, :]

    # Take the weights over the gradient's possible values
    weights = np.mean(grads_val, axis=0)

    # CAM = Class Activation Map
    cam = np.zeros(output.shape, dtype=np.float32)
    # Weight the values of the filter by the classifier's values
    for i in range(len(weights)):
        cam[:, i] += weights[i] * output[:, i]

    # Take the positive and negative components of the CAM
    cam_pos = np.maximum(cam, 0)
    cam_neg = np.minimum(cam, 0)
    heatvals = cam_pos / np.max(cam_pos)
    coldvals = -1 *  cam_neg / np.min(cam_neg)

    # Compile final heat and cold map
    heatmap = np.zeros(image_size)
    coldmap = np.zeros(image_size)

    for i in range(image_size-kernel_size):
        heatmap[i:kernel_size+i] += np.sum[heatvals[i, :]]
        coldmap[i:kernel_size+i] += np.sum[coldvals[i, :]]

    # Normalize contributions
    for i in range(kernel_size,image_size-kernel_size):
        heatmap[i] /= kernel_size
        coldmap[i] /= kernel_size

    # Handle contributions at edges
    for i in range(kernel_size):
        heatmap[i] /= (i+1)
        heatmap[image_size-i] /= (i+1)

        coldmap[i] /= (i+1)
        coldmap[image_size-i] /= (i+1)

    return heatmap, coldmap