"""

Helper functions to take the gradient of a CNN

"""

import numpy as np
import keras.backend as K


def model_gradient(model, X: np.array, output_index = 0,
                   ret_output: bool = False):
    """
    Returns the gradient of a given component of the output at
    output index of a model.
    :param model: Keras model
    :param X: Input vector to take gradient w.r.t.
    :param output_index: Which output index of the CNN to use
    :param ret_output: Return output or not
    :return:
    """


    inputs = model.input
    outputs = model.output

    focus = outputs[0][output_index]

    grad = K.gradients(focus,inputs)[0]

    if ret_output:
        f = K.function([inputs, [outputs, grad]])
        return f([X])
    else:
        f = K.function([inputs, grad])
        return f([X])

def fd_gradient(model, x0, dx=.005, order=4,
                ret_output = False):
    """
    Take the finite-difference gradient of a NN
    in order to sanity-check against the model gradient function above.
    :param model: Keras model
    :param x0: Input vector to differentiate w.r.t.
    :param dx: Steps size
    :param order: order of the finite difference function
    :param ret_output: Return the output of the NN as well as the gradient
    :return:
    """
    gradient = []

    y0 = model.predict(x0)

    zero = np.zeros(shape=x0.shape)
    # See: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # Second order finite difference stencil
    if order == 2:
        for i, x in enumerate(x0[0]):
            zp = np.copy(zero)
            zp[0][i] = dx
            yp = model.predict(x0 + zp)
            ym = model.predict(x0 - zp)

            gradient.append((yp - ym) / (2 * dx))
    # Fourth order: Less numerical noise introduced
    elif order == 4:
        for i, x in enumerate(x0[0]):
            zp = np.copy(zero)
            zp[0][i] = dx
            ypp = model.predict(x0 + 2 * zp)
            yp = model.predict(x0 + zp)
            ym = model.predict(x0 - zp)
            ymm = model.predict(x0 - 2 * zp)
            gradient.append((-ypp / 4 + 2 * yp - 2 * ym + ymm / 4) / (3 * dx))

    if ret_output:
        return y0, np.array(gradient)
    else:
        return np.array(gradient)


def plot_model_gradient(model, X, curve_color,):
    raise NotImplementedError