from dl_backtrace.tf_backtrace import Backtrace as TFBacktrace
from dl_backtrace.pytorch_backtrace import Backtrace as TorchBacktrace
import tensorflow as tf
import torch
import numpy as np

def backtrace_quantus(model, inputs, targets, **kwargs) -> np.ndarray:
    mode_name = kwargs['mode']
    cmode = None
    if kwargs['cmode'] is not None and mode_name == 'contrast':
        cmode = kwargs['cmode']
    torch_model = False
    if isinstance(model, tf.keras.Model):  # TensorFlow model
        backtrace = TFBacktrace(model=model)
    elif isinstance(model, torch.nn.Module):  # PyTorch model
        backtrace = TorchBacktrace(model=model)
        torch_model=True
    return backtrace_dl_explain(model, inputs, targets, backtrace, mode_name,torch_model,cmode)

def backtrace_dl_explain(model, inputs, targets, backtrace,mode_name,torch_model,cmode):
    batch_relevance = []
    if torch_model:
        if cmode !=None : 
            for i in range(inputs.shape[0]):
                np_array = np.expand_dims(inputs[i], axis=0)
                torch_tensor = torch.from_numpy(np_array)
                layer_outputs = backtrace.predict(torch_tensor)
                relevance = backtrace.eval(layer_outputs, mode=mode_name)
                batch_relevance.append(relevance[list(relevance.keys())[-1]][cmode])
        else: 
            for i in range(inputs.shape[0]):
                np_array = np.expand_dims(inputs[i], axis=0)
                torch_tensor = torch.from_numpy(np_array)
                layer_outputs = backtrace.predict(torch_tensor)
                relevance = backtrace.eval(layer_outputs, mode=mode_name)
                batch_relevance.append(relevance[list(relevance.keys())[-1]])
        batch_relevance = np.array(batch_relevance)
                
    else:
        if cmode !=None : 
            for i in range(inputs.shape[0]):
                np_array = np.expand_dims(inputs[i], axis=0)
                layer_outputs = backtrace.predict(np_array)
                relevance = backtrace.eval(layer_outputs, mode=mode_name)
                batch_relevance.append(relevance[list(relevance.keys())[-1]][cmode])
        else :
            for i in range(inputs.shape[0]):
                np_array = np.expand_dims(inputs[i], axis=0)
                layer_outputs = backtrace.predict(np_array)
                relevance = backtrace.eval(layer_outputs, mode=mode_name)
                batch_relevance.append(relevance[list(relevance.keys())[-1]])

        batch_relevance = np.array(batch_relevance)

    return batch_relevance
