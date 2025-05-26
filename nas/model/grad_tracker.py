import torch
import torch.nn as nn

from performer_pytorch import SelfAttention

class GradTracker():
    def __init__(self, model, register_modules=None):
        self.model = model
        self.register_modules = {}
        self.module_input = {}
        self.module_output = {}
        self.module_grad_input = {}
        self.module_grad_output = {}
        self.handles = []
        
        if register_modules is None:
            self.get_register_modules()
        else:
            self.register_modules = register_modules
        self.register_hook()
    
    def get_register_modules(self):
        for layer in self.model.layers:
            layer_modules = []
            for local_model in layer.local_models:
                if local_model is not None:
                    layer_modules.append(local_model)
            for self_attn in layer.self_attns:
                if self_attn is not None:
                    layer_modules.append(self_attn)
            self.register_modules.append(layer_modules)
    
    def register_hook(self):
        for layer_modules in self.register_modules:
            for module in layer_modules:
                handle = module.register_forward_hook(self.forward_hook)
                self.handles.append(handle)
                # handle = module.register_backward_hook(self.backward_hook)
                handle = module.register_full_backward_hook(self.backward_hook)
                self.handles.append(handle)
                
    def forward_hook(self, module, input, output):
        self.module_input[module] = input
        if isinstance(module, torch.nn.MultiheadAttention):
            self.module_output[module] = output[0]
        else:
            self.module_output[module] = output
    
    def backward_hook(self, module, grad_input, grad_output):
        self.module_grad_input[module] = grad_input
        self.module_grad_output[module] = grad_output
        
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []