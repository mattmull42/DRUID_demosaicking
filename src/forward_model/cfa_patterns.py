"""A file containing all the CFA patterns and related functions.
"""

import torch


RED_FILTER = torch.Tensor([1, 0, 0])
GREEN_FILTER = torch.Tensor([0, 1, 0])
BLUE_FILTER = torch.Tensor([0, 0, 1])
PAN_FILTER = torch.Tensor([1 / 3, 1 / 3, 1 / 3])


def get_bayer_GRBG_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((2, 2, 1)), GREEN_FILTER)

    pattern[0, 1] = RED_FILTER
    pattern[1, 0] = BLUE_FILTER

    return pattern


def get_bayer_RGGB_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((2, 2, 1)), GREEN_FILTER)

    pattern[0, 0] = RED_FILTER
    pattern[1, 1] = BLUE_FILTER

    return pattern


def get_quad_bayer_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), GREEN_FILTER)

    pattern[0, 2] = RED_FILTER
    pattern[0, 3] = RED_FILTER
    pattern[1, 2] = RED_FILTER
    pattern[1, 3] = RED_FILTER

    pattern[2, 0] = BLUE_FILTER
    pattern[2, 1] = BLUE_FILTER
    pattern[3, 0] = BLUE_FILTER
    pattern[3, 1] = BLUE_FILTER

    return pattern


def get_sparse_3_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((8, 8, 1)), PAN_FILTER)

    pattern[0, 0] = RED_FILTER

    pattern[0, 4] = GREEN_FILTER
    pattern[4, 0] = GREEN_FILTER

    pattern[4, 4] = BLUE_FILTER

    return pattern


def get_kodak_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), PAN_FILTER)

    pattern[3, 2] = RED_FILTER
    pattern[2, 3] = RED_FILTER

    pattern[3, 0] = GREEN_FILTER
    pattern[2, 1] = GREEN_FILTER
    pattern[1, 2] = GREEN_FILTER
    pattern[0, 3] = GREEN_FILTER

    pattern[1, 0] = BLUE_FILTER
    pattern[0, 1] = BLUE_FILTER

    return pattern


def get_sony_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), PAN_FILTER)

    pattern[2, 3] = RED_FILTER
    pattern[0, 1] = RED_FILTER

    pattern[3, 0] = GREEN_FILTER
    pattern[2, 1] = GREEN_FILTER
    pattern[1, 2] = GREEN_FILTER
    pattern[0, 3] = GREEN_FILTER

    pattern[3, 2] = BLUE_FILTER
    pattern[1, 0] = BLUE_FILTER

    return pattern


def get_chakrabarti_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((6, 6, 1)), PAN_FILTER)

    pattern[2, 3] = RED_FILTER
    pattern[2, 2] = GREEN_FILTER
    pattern[3, 3] = GREEN_FILTER
    pattern[3, 2] = BLUE_FILTER

    return pattern


def get_honda_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), PAN_FILTER)

    pattern[1, 3] = RED_FILTER

    pattern[1, 1] = GREEN_FILTER
    pattern[3, 3] = GREEN_FILTER

    pattern[3, 1] = BLUE_FILTER

    return pattern


def get_honda2_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((2, 2, 1)), PAN_FILTER)

    pattern[1, 0] = RED_FILTER
    pattern[0, 1] = GREEN_FILTER
    pattern[0, 0] = BLUE_FILTER

    return pattern


def get_kaizu_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((6, 6, 1)), PAN_FILTER)

    pattern[0, 0] = RED_FILTER
    pattern[1, 1] = RED_FILTER
    pattern[4, 2] = RED_FILTER
    pattern[5, 3] = RED_FILTER
    pattern[2, 4] = RED_FILTER
    pattern[3, 5] = RED_FILTER

    pattern[0, 2] = GREEN_FILTER
    pattern[1, 3] = GREEN_FILTER
    pattern[2, 0] = GREEN_FILTER
    pattern[3, 1] = GREEN_FILTER
    pattern[4, 4] = GREEN_FILTER
    pattern[5, 5] = GREEN_FILTER

    pattern[4, 0] = BLUE_FILTER
    pattern[5, 1] = BLUE_FILTER
    pattern[2, 2] = BLUE_FILTER
    pattern[3, 3] = BLUE_FILTER
    pattern[0, 4] = BLUE_FILTER
    pattern[1, 5] = BLUE_FILTER

    return pattern


def get_yamagami_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), PAN_FILTER)

    pattern[0, 2] = RED_FILTER
    pattern[2, 0] = RED_FILTER

    pattern[1, 1] = GREEN_FILTER
    pattern[1, 3] = GREEN_FILTER
    pattern[3, 1] = GREEN_FILTER
    pattern[3, 3] = GREEN_FILTER

    pattern[0, 0] = BLUE_FILTER
    pattern[2, 2] = BLUE_FILTER

    return pattern


def get_gindele_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((2, 2, 1)), GREEN_FILTER)

    pattern[0, 1] = RED_FILTER
    pattern[1, 0] = BLUE_FILTER
    pattern[1, 1] = PAN_FILTER

    return pattern


def get_hamilton_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((8, 8, 1)), PAN_FILTER)

    pattern[0, 0] = RED_FILTER
    pattern[1, 1] = RED_FILTER
    pattern[2, 2] = RED_FILTER
    pattern[3, 3] = RED_FILTER
    pattern[2, 0] = RED_FILTER
    pattern[3, 1] = RED_FILTER
    pattern[0, 2] = RED_FILTER
    pattern[1, 3] = RED_FILTER

    pattern[0, 4] = GREEN_FILTER
    pattern[0, 6] = GREEN_FILTER
    pattern[1, 5] = GREEN_FILTER
    pattern[1, 7] = GREEN_FILTER
    pattern[2, 4] = GREEN_FILTER
    pattern[2, 6] = GREEN_FILTER
    pattern[3, 5] = GREEN_FILTER
    pattern[3, 7] = GREEN_FILTER

    pattern[4, 0] = GREEN_FILTER
    pattern[6, 0] = GREEN_FILTER
    pattern[5, 1] = GREEN_FILTER
    pattern[7, 1] = GREEN_FILTER
    pattern[4, 2] = GREEN_FILTER
    pattern[6, 2] = GREEN_FILTER
    pattern[5, 3] = GREEN_FILTER
    pattern[7, 3] = GREEN_FILTER

    pattern[4, 4] = BLUE_FILTER
    pattern[5, 5] = BLUE_FILTER
    pattern[6, 6] = BLUE_FILTER
    pattern[7, 7] = BLUE_FILTER
    pattern[6, 4] = BLUE_FILTER
    pattern[7, 5] = BLUE_FILTER
    pattern[4, 6] = BLUE_FILTER
    pattern[5, 7] = BLUE_FILTER

    return pattern


def get_luo_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), PAN_FILTER)

    pattern[1, 0] = RED_FILTER
    pattern[1, 2] = RED_FILTER

    pattern[0, 1] = GREEN_FILTER
    pattern[2, 1] = GREEN_FILTER

    pattern[1, 1] = BLUE_FILTER

    return pattern


def get_wang_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((5, 5, 1)), PAN_FILTER)

    pattern[2, 0] = RED_FILTER
    pattern[0, 1] = RED_FILTER
    pattern[1, 3] = RED_FILTER
    pattern[3, 2] = RED_FILTER
    pattern[4, 4] = RED_FILTER

    pattern[3, 0] = GREEN_FILTER
    pattern[1, 1] = GREEN_FILTER
    pattern[4, 2] = GREEN_FILTER
    pattern[2, 3] = GREEN_FILTER
    pattern[0, 4] = GREEN_FILTER

    pattern[4, 0] = BLUE_FILTER
    pattern[2, 1] = BLUE_FILTER
    pattern[0, 2] = BLUE_FILTER
    pattern[3, 3] = BLUE_FILTER
    pattern[1, 4] = BLUE_FILTER

    return pattern


def get_yamanaka_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), GREEN_FILTER)

    pattern[0, 1] = RED_FILTER
    pattern[1, 3] = RED_FILTER
    pattern[2, 1] = RED_FILTER
    pattern[3, 3] = RED_FILTER

    pattern[1, 1] = BLUE_FILTER
    pattern[0, 3] = BLUE_FILTER
    pattern[3, 1] = BLUE_FILTER
    pattern[2, 3] = BLUE_FILTER

    return pattern


def get_lukac_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), GREEN_FILTER)

    pattern[0, 1] = RED_FILTER
    pattern[2, 0] = RED_FILTER
    pattern[0, 3] = RED_FILTER
    pattern[2, 2] = RED_FILTER

    pattern[1, 1] = BLUE_FILTER
    pattern[3, 0] = BLUE_FILTER
    pattern[1, 3] = BLUE_FILTER
    pattern[3, 2] = BLUE_FILTER

    return pattern


def get_xtrans_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((6, 6, 1)), GREEN_FILTER)

    pattern[0, 4] = RED_FILTER
    pattern[1, 0] = RED_FILTER
    pattern[1, 2] = RED_FILTER
    pattern[2, 4] = RED_FILTER
    pattern[3, 1] = RED_FILTER
    pattern[4, 3] = RED_FILTER
    pattern[4, 5] = RED_FILTER
    pattern[5, 1] = RED_FILTER

    pattern[0, 1] = BLUE_FILTER
    pattern[1, 3] = BLUE_FILTER
    pattern[1, 5] = BLUE_FILTER
    pattern[2, 1] = BLUE_FILTER
    pattern[3, 4] = BLUE_FILTER
    pattern[4, 0] = BLUE_FILTER
    pattern[4, 2] = BLUE_FILTER
    pattern[5, 4] = BLUE_FILTER

    return pattern


def get_binning_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((4, 4, 1)), PAN_FILTER)

    pattern[0, 0] = GREEN_FILTER
    pattern[0, 1] = GREEN_FILTER
    pattern[1, 0] = GREEN_FILTER
    pattern[1, 1] = GREEN_FILTER

    pattern[0, 2] = RED_FILTER
    pattern[0, 3] = RED_FILTER
    pattern[1, 2] = RED_FILTER
    pattern[1, 3] = RED_FILTER

    pattern[2, 0] = BLUE_FILTER
    pattern[2, 1] = BLUE_FILTER
    pattern[3, 0] = BLUE_FILTER
    pattern[3, 1] = BLUE_FILTER

    return pattern


def get_random_pattern() -> torch.Tensor:
    pattern = torch.kron(torch.ones((8, 8, 1)), PAN_FILTER)

    pattern[1, 7] = RED_FILTER
    pattern[3, 1] = RED_FILTER
    pattern[3, 3] = RED_FILTER
    pattern[5, 7] = RED_FILTER
    pattern[7, 1] = RED_FILTER

    pattern[1, 3] = GREEN_FILTER
    pattern[3, 7] = GREEN_FILTER
    pattern[5, 3] = GREEN_FILTER
    pattern[5, 5] = GREEN_FILTER
    pattern[7, 7] = GREEN_FILTER

    pattern[1, 1] = BLUE_FILTER
    pattern[1, 5] = BLUE_FILTER
    pattern[3, 5] = BLUE_FILTER
    pattern[5, 1] = BLUE_FILTER
    pattern[7, 3] = BLUE_FILTER
    pattern[7, 5] = BLUE_FILTER

    return pattern
