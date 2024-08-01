"""The file describing the CFA operator.
"""

import torch

from . import cfa_patterns


class cfa_operator():
    def __init__(self, cfa: str, input_shape: tuple) -> None:
        """Creates an instane of the cfa_operator class.

        Args:
            cfa (str): The name of the CFA to be used.
            input_shape (tuple): The shape of the object the operator takes in input.
        """
        self.cfa = cfa
        self.pattern = getattr(cfa_patterns, f'get_{cfa}_pattern')()
        self.pattern_shape = self.pattern.shape
        self.input_shape = input_shape
        self.output_shape = input_shape[:-1]

        n = input_shape[0] // self.pattern_shape[0] + (input_shape[0] % self.pattern_shape[0] != 0)
        m = input_shape[1] // self.pattern_shape[1] + (input_shape[1] % self.pattern_shape[1] != 0)

        self.mask = torch.tile(self.pattern, (n, m, 1))[:input_shape[0], :input_shape[1]]

    def direct(self, x: torch.Tensor) -> torch.Tensor:
        """A method method performing the computation of the operator.

        Args:
            x (torch.Tensor): The input array. Must be of shape self.input_shape.

        Returns:
            torch.Tensor: The output array. Must be of shape self.output_shape.
        """
        return torch.sum(x * self.mask, axis=2)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """A method performing the computation of the adjoint of the operator.

        Args:
            y (torch.Tensor): The input array. Must be of shape self.output_shape.

        Returns:
            torch.Tensor: The output array. Must be of shape self.input_shape.
        """
        return self.mask * y[..., None]
