import tensorflow as tf
import keras.backend as K
import numpy as np
import keras
from keras.layers.core import Lambda


def _target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def _target_category_loss_output_shape(input_shape):
    return input_shape

def _normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(input_model, image, category_index,failure_warning = False,
             nb_classes = 3,kernel_size = 8):
    # V important: make sure that the category index is the same for all examples
    # Category index is 0, 1, 2

    if len(image.shape) == 23:
        image = image.reshape(image.shape[0], image.shape[1], 1)

    # Clone a model to generate grad-cam numbers with
    model = keras.models.clone_model(input_model)
    model.set_weights(input_model.get_weights())

    # How many classes are used?
    # generate a new layer to 0 out contributions from other classes
    target_layer = lambda x: _target_category_loss(x, category_index,
                                                  nb_classes)
    model.add(Lambda(target_layer,
                     output_shape=_target_category_loss_output_shape))

    # The classification score is therefore
    #  only the output from this final layer
    loss = K.sum(model.layers[-1].output)

    # What is the final convolutional layer?
    # Here, the first layer. This points to the output
    conv_output = model.layers[0].output

    # Gradients wrt to the convolution layer's output
    grads = _normalize(K.gradients(loss, conv_output)[0])
    # Turn this into a keras function
    gradient_function = K.function([model.layers[0].input],
                                   [conv_output, grads])
    # Now compute the output of the convolutional layer and
    #  the gradients of the loss wrt them
    output, grads_val = gradient_function([image])

    # Mean weights of each kernel
    weights = np.mean(grads_val, axis=1)

    # CAM = Class Activation Map
    cam = np.zeros(output.shape, dtype=np.float32)

    for i in range(cam.shape[0]):
        for j in range(weights.shape[1]):
            cam[i, :, j] += weights[i, j] * output[i, :, j]

    cam = np.maximum(cam, 0)
    negcam = np.minimum(cam, 0)

    if not np.any(cam) and failure_warning:
        print("WARNING! Grad cam heat map procedure failed; all weighted "
              "activation was negative.")
        heatmap = cam
    else:
        heatmap = cam / np.max(cam)


    if not np.any(negcam) and failure_warning:
        print("WARNING! Grad cam cold map procedure failed; all weighted "
              "activation was positive.")
        coldmap = negcam
    else:
        coldmap = negcam / np.min(negcam)


    heat_domain = np.zeros((cam.shape[0], 100))
    cold_domain = np.zeros((cam.shape[0], 100))

    for j in range(cam.shape[0]):

        for i in range(100-kernel_size+1):
            heat_domain[j, i:kernel_size + i] += np.sum(heatmap[j, i, :])
            cold_domain[j, i:kernel_size + i] += np.sum(coldmap[j, i, :])

        for i in range(kernel_size):
            heat_domain[j, i] /= (i + 1)
            heat_domain[j, 99 - i] /= (i + 1)
            cold_domain[j, i] /= (i + 1)
            cold_domain[j, 99 - i] /= (i + 1)

        for i in range(kernel_size, 100-kernel_size):
            heat_domain[j, i] /= kernel_size
            cold_domain[j, i] /= kernel_size

        heat_domain[j, :] /= np.max(heat_domain[j, :])

    return np.uint8(cam), heatmap, heat_domain, cold_domain