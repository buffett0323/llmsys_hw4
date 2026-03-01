"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        
        # COPY FROM ASSIGN2_3
        self.weights = Parameter(tensor_from_numpy(np.random.normal(0, 1, (num_embeddings, embedding_dim)), backend=backend))
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        
        # COPY FROM ASSIGN2_3
        x_flat = x.contiguous().view(bs * seq_len)
        one_hot_flat = one_hot(x_flat, self.num_embeddings)
        out_flat = one_hot_flat @ self.weights.value
        return out_flat.view(bs, seq_len, self.embedding_dim)

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        # COPY FROM ASSIGN2_3
        if not self.training:
            return x
        if self.p_dropout == 0:
            return x
        # To match the autograder seed, please use np.random.binomial to generate a mask.
        # mask: 1 = keep, 0 = drop. binomial(n=1, p=1-p_dropout) gives keep prob 1-p_dropout
        mask = tensor_from_numpy(
            np.random.binomial(1, 1 - self.p_dropout, size=x.shape).astype(np.float32),
            backend=x.backend,
        )
        return x * mask / (1 - self.p_dropout)


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.out_size = out_size
        
        # COPY FROM ASSIGN2_3
        self.bias_flag = bias
        ### BEGIN ASSIGN3_2
        # raise NotImplementedError
        self.weights = Parameter(tensor_from_numpy(np.random.uniform(-1/np.sqrt(in_size), 1/np.sqrt(in_size), (in_size, out_size)), backend=backend))
        if bias:
            self.bias = Parameter(tensor_from_numpy(np.random.uniform(-1/np.sqrt(in_size), 1/np.sqrt(in_size), (out_size,)), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        # batch, in_size = x.shape
        
        # COPY FROM ASSIGN2_3
        leading_dims = x.shape[:-1]
        in_size = x.shape[-1]
        flattened_dim = np.prod(leading_dims).item()
        
        ### BEGIN ASSIGN3_2
        # raise NotImplementedError
        x = x.contiguous()
        x_reshaped = x.view(flattened_dim, in_size)
        output = x_reshaped @ self.weights.value
        if self.bias_flag:
            output = output + self.bias.value
        output = output.contiguous()
        return output.view(*leading_dims, self.out_size)
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend, use_fused_kernel: bool = False):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
            backend : Backend for tensor operations.
            use_fused_kernel : If True, use fused CUDA layernorm kernel; else use basic ops.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        self.use_fused_kernel = use_fused_kernel
        
        # COPY FROM ASSIGN2_3
        self.weights = Parameter(tensor_from_numpy(np.ones(dim), backend=backend))
        self.bias = Parameter(tensor_from_numpy(np.zeros(dim), backend=backend))

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        x = x.view(batch, dim)
        weights = self.weights.value.view(1, dim)
        bias = self.bias.value.view(1, dim)

        if self.use_fused_kernel:
            from .tensor_functions import LayerNorm
            # LayerNorm expects gamma, beta of shape (dim,)
            gamma = self.weights.value
            beta = self.bias.value
            output = LayerNorm.apply(x, gamma, beta)
        else:
            # COPY FROM ASSIGN2_3 - basic ops implementation
            output = (x - x.mean(dim=1)) / (x.var(dim=1) + self.eps) ** 0.5 * weights + bias.view(1, dim)

        return output.view(batch, dim)
        
