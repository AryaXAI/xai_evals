from dl_backtrace.tf_backtrace import Backtrace as TFBacktrace
from dl_backtrace.pytorch_backtrace import Backtrace as TorchBacktrace
import tensorflow as tf
import torch
import numpy as np

def backtrace_quantus(model, inputs, targets, **kwargs) -> np.ndarray:
    mode_name = kwargs['mode']
    torch_model = False
    if isinstance(model, tf.keras.Model):  # TensorFlow model
        backtrace = TFBacktrace(model=model)
    elif isinstance(model, torch.nn.Module):  # PyTorch model
        backtrace = TorchBacktrace(model=model)
        torch_model=True
    return backtrace_dl_explain(model, inputs, targets, backtrace, mode_name,torch_model)

def backtrace_dl_explain(model, inputs, targets, backtrace,mode_name,torch_model):
    batch_relevance = []
    if torch_model:
        for i in range(inputs.shape[0]):
            np_array = np.expand_dims(inputs[i], axis=0)
            torch_tensor = torch.from_numpy(np_array)
            layer_outputs = backtrace.predict(torch_tensor)
            relevance = backtrace.eval(layer_outputs, mode=mode_name)
            batch_relevance.append(relevance[list(relevance.keys())[-1]])
    else:
        for i in range(inputs.shape[0]):
            np_array = np.expand_dims(inputs[i], axis=0)
            layer_outputs = backtrace.predict(np_array)
            relevance = backtrace.eval(layer_outputs, mode=mode_name)
            batch_relevance.append(relevance[list(relevance.keys())[-1]])
    return np.array(batch_relevance)
