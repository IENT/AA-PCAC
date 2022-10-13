import numpy as np


def quantize(x, qstep):
    # Midrise quantization
    x_quant_idx = np.floor(x/qstep + 0.5).astype(np.int64)
    # x_quant_idx = np.round(x/qstep).astype(np.int64)

    x_quant = dequantize(x_quant_idx, qstep)

    return x_quant, x_quant_idx


def dequantize(x_quant_idx, qstep):
    
    x_quant = x_quant_idx * qstep

    return x_quant
