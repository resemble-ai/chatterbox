import torch
import torch.nn as nn


def get_intmeanflow_time_mixer(dims):
    """"
    Diagonal init as described in 3.3 https://arxiv.org/pdf/2510.07979
    """
    layer = nn.Linear(dims * 2, dims, bias=False)

    with torch.no_grad():
        target_weight = torch.zeros(dims, 2 * dims)
        target_weight[:, 0:dims] = torch.eye(dims)
        layer.weight.data = target_weight

    return layer

if __name__ == '__main__':

    D_example = 6

    W_layer = get_intmeanflow_time_mixer(D_example)

    print(f"Layer weight (AFTER init):\n{W_layer.weight.data}\n")

    e_t = torch.tensor([0., 1., 2., 3., 4., 5.])
    e_r = torch.tensor([6., 7., 8., 9., 10., 11.])
    e_concat = torch.cat([e_t, e_r]).unsqueeze(0)  # Shape (1, 12)

    output = W_layer(e_concat)

    print(f"Test Input e_t: \n{e_t}")
    print(f"Test Input e_r: \n{e_r}")
    print(f"Test Input concat: \n{e_concat}")

    print(f"Forward Pass Output: \n{output.squeeze(0)}")
