

def get_desired_parameter_names(model, desired_layer_type):
    """
    Returns the names of the model parameters that are inside a desired layer type.
    """
    result = []
    for name, module in model.named_modules():
        if isinstance(module, tuple(desired_layer_type)):
            tmp_results = [f'{name}.{n}' for n in module._parameters.keys()]
            result += tmp_results
    return result

def get_parameter_name_grouped(model):
    ...
