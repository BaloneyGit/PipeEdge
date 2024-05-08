from typing import List, Optional, Tuple, Union
import math
import numpy as np
import torch
import monitoring
from pipeedge import models
from pipeedge.quantization.clamp_op import clamp_banner2019_gelu, clamp_banner2019_laplace
from pipeedge.quantization.basic_op import (
    compression_factor, tensor_encode_outerdim, tensor_decode_outerdim
)
from scipy.special import lambertw

MONITORING_KEY_QUANT_ENCODE = 'quant_encode'

def _quant_op(input_data, bit, mode='original'): ## TODO; create for new quant op (compute w_adptv)
    """
    The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    assert bit > 0
    assert np.all(input_data >= 0)
    assert np.all(input_data <= 1)


    # input should be in [0,1]
    # the res can be removed for further speed/memory improvement
    # if mode == 'original':
    #     scale = (1 << bit) - 1
    #     res = np.around(scale * input_data)
    #     int_map = res.copy()
    #     int_map = int_map.astype(np.uint32)
    #     res /= scale
    # elif mode == 'modified':
    #     scale = 1 << bit
    #     res = np.floor(scale * input_data)
    #     int_map = res.copy()
    #     int_map = int_map.astype(np.uint32)
    #     np.clip(res, 0, scale-1, res)
    #     res /= scale
    # else:
    #     raise ValueError('mode should be either [original] or [modified]')

    scale = (1 << bit) - 1  ## TODO: check if shifting is correct here
    res = np.around(scale * input_data)
    int_map = res.copy()
    int_map = int_map.astype(np.uint32)
    res /= scale

    assert np.all(res >= 0)
    assert np.all(res <= 1)
    return res, int_map


### AdaptFloat implementation
def get_exp_max(w_fp): ## TODO: need to compute per layer granularity normalized exp_max 
                   ## considering given constraint in paper
    """returns normalized exp_max for max(w_abs)"""

    # assert bit > 0
    # assert np.all(input_data >= 0)
    # assert np.all(input_data <= 1)

    w_abs = torch.abs(w_fp)

    exp_max = math.log(torch.max(w_abs) - 2)
    assert exp_max > math.log(torch.max(w_abs) - 2) - 1

    return exp_max, w_abs

### AdaptFloat implementation
def get_exp_bias(bits, w_fp, e): # TODO: need to compute per layer granularity exp_bias
    """returns exp_bias (the scaling factor for AdaptivFloat),
        val_max (matrix of new max value for datapt),
        val_min (matrix of new min value for datapt)
    params:
    bits : number of bits
    exp :  number of exponenents
    w_fp : full floating point weight matrix 
    """
    exp_max, w_abs = get_exp_max(w_fp)

    # # mantissa matrix
    # m_mat = []
    # for i, j in bits_mat, e_mat:
    #     m = bits_mat[i] - e_mat[j] - 1
    #     m_mat.append(m)
    # m_mat = torch.tensor(m_mat)

    # number of mantissa bits
    mants = bits - e - 1

    w_sign = torch.sign(w_fp)

    # exp_bias computation
    exp_bias = exp_max - (2**math.e - 1)
    
    # # matrix of minimum values
    # val_min_mat = []
    # for i in range(m_mat.shape(0)):
    #     val_min = 2**exp_bias * (1 + 2**(-m))
    #     val_min_mat.append(val_min)
    # val_min_mat = torch.tensor(val_min_mat)\

    val_min = 2**exp_bias * (1 + (1/2**mants))

    val_max = 2**exp_max * (2 - (1/2**mants))

    # # matrix of maximum values
    # val_max_mat = []
    # for i in range(m_mat.shape(0)):
    #     val_max = 2**exp_bias * (2 - 2**(-m))
    #     val_max_mat.append(val_max)
    # val_max_mat = torch.tensor(val_max_mat)

    # rounding and clamping
    w_abs_clamped = torch.clamp(w_abs, min=val_min, max=val_max)

    # matrices of exponents
    w_exp = np.floor(np.log2(w_abs_clamped))

    # matrix of mantissas
    w_mant = w_abs_clamped / (2 ** w_exp)

    # quantized and scaled
    w_q = w_mant * (1/2**mants)

    # final output matrix
    w_adptv = w_sign * 2**(w_exp) * w_q

    return w_adptv, val_min, val_max

def _intmap_encode(int_map, bitwidth):
    """ compress the converted int_map to tesnor with fewer numbers"""
    # the int_map is assumed as a 4- or 3-dimensional np.array [b(optional),c,h,w]
    int_map = int_map.flatten()
    # enc_ratio is the number of original values compressed into one single int32 value
    enc_ratio = int(32/bitwidth)

    # store tensor into new_tensor
    # e.g. original tensor with 6 values: [0,1,2,3,4,5] (dtype=int32)
    # new tensor with 2 values: [3,2,1,0], [5,4,NULL,NULL] (enc_ratio=4, one int32 has 4 values)
    int_map_ext = np.append(int_map,
                            np.repeat(0, (enc_ratio - len(int_map) % enc_ratio) % enc_ratio))
    int_map_rs = np.reshape(int_map_ext, (-1, enc_ratio))
    bitshift = np.array([(i % enc_ratio) * bitwidth for i in range(enc_ratio)], dtype=np.uint32)
    int_map_shifted = np.left_shift(int_map_rs, bitshift)
    new_array = np.bitwise_or.reduce(int_map_shifted, axis=1, dtype=np.uint32)

    return new_array

def _uint32_to_uint8(tensor):
    """ re-represent uint32 to uint8, since torch has no uint32 (does have uint8) """
    assert tensor.dtype == np.uint32
    return tensor.view('uint8')

def tensor_encode(input_data: torch.Tensor, quant_bit: int) -> List[torch.Tensor]:
    """
        The input to the encoder should be a torch.Tensor
        We first cast it to a np.array, then do everything else
    """
    quant_bit_tensor = torch.tensor(quant_bit, dtype = torch.int8)
    if quant_bit == 0:
        return [input_data, torch.tensor(input_data.shape), torch.tensor(1.0), torch.tensor(0.0),
                quant_bit_tensor]

    input_data = input_data.numpy()
    shape = input_data.shape
    # ensure the input is scaled to [0,1],
    shift = input_data.min()
    input_data = input_data - shift
    scale_factor = input_data.max()
    rescale_input = input_data/scale_factor
    # quant
    _, int_map = _quant_op(rescale_input, quant_bit)
    comm_tensor = _intmap_encode(int_map, quant_bit)
    # split uint32 into 4 uint8
    comm_tensor = _uint32_to_uint8(comm_tensor)
    # convert array to tensor for p2p communication
    comm_tensor = torch.tensor(comm_tensor, dtype = torch.uint8)
    shape = torch.tensor(shape, dtype = torch.int32)
    scale_factor = torch.tensor(scale_factor, dtype = torch.float32)
    shift = torch.tensor(shift, dtype = torch.float32)

    # scale_factor is needed to restore the tensor
    return [comm_tensor, shape, scale_factor, shift, quant_bit_tensor]

def _clamp_factor_laplace(bit: int) -> torch.Tensor:
    # scipy returns a float64, but we'll overflow first if we don't force it when bit>=32
    return lambertw(3 * torch.tensor(4, dtype=torch.float64)**bit).real

def clamp_banner2019_laplace(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Clamp tensor with a Laplace distribution - based on Banner et. al.'s NIPS 2019 paper."""
    # "Post training 4-bit quantization of convolutional networks for rapid-deployment"
    variance = torch.var(tensor, unbiased = False)
    dist_parameter = torch.sqrt(0.5*variance)
    alpha = _clamp_factor_laplace(bit).to(tensor) * dist_parameter
    return tensor.clamp(min=-alpha, max=alpha)

def _clamp_factor_gelu(bit: int) -> torch.Tensor:
    # scipy returns a float64, but we'll overflow first if we don't force it when bit>=31
    return lambertw(3 * torch.tensor(4, dtype=torch.float64)**(bit+1)).real

def clamp_banner2019_gelu(tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Like `clamp_banner2019_laplace` but modified for a GeLU layer output."""
    # Special case for GeLU layer
    # Distribution after GeLU only has half of bell curve
    # Assuming mean = 0, and ignore the influence of negtive small values
    variance = 2* torch.pow(tensor, 2).sum()/torch.numel(tensor)
    dist_parameter = torch.sqrt(0.5*variance)
    alpha = _clamp_factor_gelu(bit).to(tensor) * dist_parameter
    return tensor.clamp(min=-alpha, max=alpha)

def tensor_encode_outerdim(batched_tensor: torch.Tensor, quant_bit: int) -> List[torch.Tensor]:
    """do quantization on each image in the micro-batched tensor with size [b,c,h,w]"""
    list_of_lists = [tensor_encode(t, quant_bit) for t in batched_tensor]
    encoded_tensors = list(zip(*list_of_lists))
    return [torch.stack(t,0) for t in encoded_tensors]


def forward_hook_quant_encode(module, _input_arg, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
    """encode tensor in the forward hook (after each module)""" 

    monitoring.iteration_start(MONITORING_KEY_QUANT_ENCODE)
    if isinstance(output, torch.Tensor):
        output = (output,)
    assert isinstance(output, tuple)
    quant_bit = module.quant_bit.item()
    comm_tuple = []
    for tensor in output:
        assert isinstance(tensor, torch.Tensor)
        if quant_bit > 0: ## TODO: create new clamp operation using val_min, val_max and exp_bias for laplace dist
            clamp = clamp_banner2019_laplace if tensor.min() < 0.2 else clamp_banner2019_gelu 
            tensor = clamp(tensor, quant_bit)
        stacked_tensor = tensor_encode_outerdim(tensor, quant_bit)
        comm_tuple += stacked_tensor
    # Measure work as the microbatch size, but quantization only does work if quant_bit > 0.
    n_items = models.get_microbatch_size(output[0], verify=True) if quant_bit > 0 else 0
    monitoring.iteration(MONITORING_KEY_QUANT_ENCODE, work=n_items, accuracy=quant_bit)
    return tuple(comm_tuple)

