from typing import Any, Callable, Dict, Union

eval_strategy_dict: Dict[str, Any] = {}
loader_dict: Dict[str, Any] = {}

def register_base(mapping: Dict[str, Any], key: str,
                  module: Any = None) -> Union[None, Callable]:
    r"""Base function for registering a module in GraphGym.

    Args:
        mapping (dict): :python:`Python` dictionary to register the module.
            hosting all the registered modules
        key (str): The name of the module.
        module (any, optional): The module. If set to :obj:`None`, will return
            a decorator to register a module.
    """
    if module is not None:
        if key in mapping:
            raise KeyError(f"Module with '{key}' already defined")
        mapping[key] = module
        return

    # Other-wise, use it as a decorator:
    def bounded_register(module):
        register_base(mapping, key, module)
        return module

    return bounded_register

def register_eval_strategy(key: str, module: Any = None):
    r"""Registers model performance evaluation strategy in nas."""
    return register_base(eval_strategy_dict, key, module)

def register_loader(key: str, module: Any = None):
    r"""Registers dataset loader in nas."""
    return register_base(loader_dict, key, module)