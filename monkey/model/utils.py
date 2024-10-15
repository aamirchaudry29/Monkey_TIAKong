import torch.nn


def get_activation_function(name: str):
    """
    Return torch.nn Activation function
    matching input name
    """

    functions = {
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
    }  # add more as needed

    name = name.lower()
    if name in functions:
        return functions[name]()
    else:
        raise ValueError(f"Undefined loss function: {name}")
