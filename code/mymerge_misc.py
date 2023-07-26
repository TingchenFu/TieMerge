import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())


import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
from src.utils.analysis_utils import *
from transformers import AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
import torch.nn.functional as F



def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )





def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def greater_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


def less_than_std_mask(tensor, factor, return_mask=False):
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() < factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)


# def topk_values_mask(M, K=0.7, return_mask=False):
#     if K > 1:
#         K /= 100

#     original_shape = M.shape
#     if M.dim() == 1:
#         M = M.unsqueeze(0)

#     n, d = M.shape
#     k = int(d * K)
#     k = d - k  # Keep top k elements instead of bottom k elements

#     # Find the k-th smallest element by magnitude for each row
#     kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
#     # Create a mask tensor with True for the top k elements in each row
#     mask = M.abs() >= kth_values
#     final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

#     if return_mask:
#         return M * final_mask, final_mask.float().mean(dim=1), final_mask
#     return M * final_mask, final_mask.float().mean(dim=1)


def bottomk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)

    # Create a mask tensor with True for the bottom k elements in each row
    mask = M.abs() <= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)




def sign_agreement_ratio(M):
    positive_count = (M > 0).float().sum(dim=0)
    negative_count = (M < 0).float().sum(dim=0)

    non_zero_values = positive_count + negative_count

    sar = torch.where(
        non_zero_values != 0,
        torch.abs(positive_count - negative_count) / non_zero_values,
        torch.ones_like(non_zero_values),
    )

    return sar


def replace_noise_and_constant(tensor, mask, replace_factor, sign_tensor):
    tensor[mask] = 0

    if replace_factor != 0:
        tensor[~mask] = replace_factor * tensor.std()
        tensor *= sign_tensor

    return tensor


def plot_topk_norms(flat_checkpoints, task_names, topks):
    check_norms = flat_checkpoints.norm(dim=1)

    all_topk_norm = []
    for topk in topks:
        print(topk)
        topk_vector, *_ = topk_values_mask(flat_checkpoints, K=topk)
        topk_norms = topk_vector.norm(dim=1)
        all_topk_norm.append(topk_norms)
    all_topk_norm = torch.vstack(all_topk_norm)
    task_topk_norms = all_topk_norm.T
    normalized_task_topk_norms = task_topk_norms / check_norms.unsqueeze(1)

    plot_rows(
        np.array(topks),
        np.array(normalized_task_topk_norms),
        labels=[f"{task_names[i]}" for i in range(len(flat_checkpoints))],
        title="Norm of top-k% parameters over total norm for each task",
        x_label="top-k%",
        y_label="Norm@k",
    )


def plot_rows(X, Y, labels=None, title=None, x_label=None, y_label=None):
    if X.shape[0] != Y.shape[1]:
        raise ValueError(
            "Length of vector X must match the number of columns in matrix Y."
        )

    if labels is not None and len(labels) != Y.shape[0]:
        raise ValueError("Number of labels must match the number of rows in matrix Y.")

    for i in range(Y.shape[0]):
        plt.plot(X, Y[i], label=labels[i] if labels is not None else None)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if labels is not None:
        plt.legend()

    plt.show()


def plot_row_histograms(M, task_names, bins=None):
    n, d = M.shape
    num_cols = min(
        n, 4
    )  # Adjust this value to change the number of columns in the subplot grid
    num_rows = (n + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

    for i, row in enumerate(M):
        ax = (
            axes[i // num_cols, i % num_cols]
            if num_rows > 1 or num_cols > 1
            else axes[i]
        )
        ax.hist(row.numpy(), bins=bins)
        ax.set_title(f"{task_names[i]}")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")

    # Remove extra subplots if any
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

    plt.tight_layout()
    plt.show()





def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return torch.sign(Tensor[norm_fracs.argmax(dim=0), torch.arange(Tensor.shape[1])])


def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()








# def tv_merging(tv_flat_checks):
#     """Merging by creating and scaling Task Vectors"""
#     all_checks = tv_flat_checks.clone()
#     tv_merged_check = aggregate(all_checks, "sum", final_signs=None)
#     return tv_merged_check


# def merge_with_oracle_sign(
#     final_signs,
#     flat_task_checks,
#     reset_thresh,
#     merge_func,
# ):
#     all_checks = flat_task_checks.clone()
#     if reset_thresh != "none":
#         logger.info(f"Pruning: {reset_thresh}")
#         updated_checks, *_ = topk_values_mask(
#             all_checks, K=reset_thresh, return_mask=False
#         )
#     else:
#         logger.info("Not removing NOISE")
#         updated_checks = all_checks
#     # updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
#     merged_tv = disjoint_aggregate(updated_checks, merge_func, final_signs)
#     return merged_tv













# def resolve_lambda_code(lambda_code):
#     if type(lambda_code) is tuple:
#         lambda_list = torch.tensor(lambda_code)
#     elif isinstance(lambda_code, float) or isinstance(lambda_code, int):
#         lambda_list = torch.tensor([lambda_code])
#     elif "linear+" in lambda_code:
#         search_lambda, start, end, step = lambda_code.split("+")
#         lambda_list = np.arange(eval(start), eval(end), eval(step))
#     elif "mergelist" in lambda_code:
#         task_lambdas = lambda_code.split("+")[-1].split(",")
#         lambda_list = np.array(task_lambdas).astype(float).tolist()
#     else:
#         raise NotImplementedError(f"Unable to decode lambda_code {lambda_code}")
#     return lambda_list

