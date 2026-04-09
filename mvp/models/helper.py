import torch.nn as nn


def get_activation_fun(activation_fun, **kwargs):
    """Helper for building an activation layer."""
    activation_fun = activation_fun.lower()
    if activation_fun == "relu":
        return nn.ReLU(**kwargs)
    elif activation_fun == "silu" or activation_fun == "swish":
        return nn.SiLU(**kwargs)
    elif activation_fun == "gelu":
        return nn.GELU(**kwargs)
    elif activation_fun == "tanh":
        return nn.Tanh()
    elif activation_fun == "selu":
        return nn.SELU(**kwargs)
    elif activation_fun == "elu":
        return nn.ELU(**kwargs)
    else:
        raise AssertionError("Unknown activation function: {}".format(activation_fun))
