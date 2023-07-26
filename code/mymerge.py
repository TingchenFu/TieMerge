import torch
from collections import OrderedDict
import os, copy
import numpy as np

def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult

def vectorize(pretrained_statedict,finetune_statedicts,remove_keys,reshape_keys):
    def state_dict_to_vector(state_dict, remove_keys=[]):
        shared_state_dict = copy.deepcopy(state_dict)
        for key in remove_keys:
            if key in shared_state_dict:
                del shared_state_dict[key]
        sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
        )



    for key in remove_keys + reshape_keys:
        pretrained_statedict[key] = pretrained_statedict[key][: len(finetune_statedicts[0][key]), :]

    finetune_vectors = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in finetune_statedicts]
    )
    pretrained_vector = state_dict_to_vector(pretrained_statedict, remove_keys)

    return pretrained_vector, finetune_vectors



def aggregate(T, agg_type, final_signs, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs

    return result

def disjoint_aggregate(Tensor, agg_mode, sign_to_mult):
    agg_mode = agg_mode.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep
    
    if agg_mode == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif agg_mode == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif agg_mode == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {agg_mode} is not defined.")
    
    return disjoint_aggs



def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict




def topk_values_mask(finetune_vectors, reset_threshold=0.7, return_mask=False):
    if reset_threshold > 1:
        reset_threshold= reset_threshold / 100
    
    original_shape = finetune_vectors.shape
    if finetune_vectors.dim() == 1:
        finetune_vectors = finetune_vectors.unsqueeze(0)

    n, d = finetune_vectors.shape
    k = int(d * reset_threshold)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = finetune_vectors.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = finetune_vectors.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == finetune_vectors.squeeze().shape else mask

    if return_mask:
        return finetune_vectors * final_mask, final_mask.float().mean(dim=1), final_mask
    return finetune_vectors * final_mask



def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
    else:
        raise NotImplementedError
    # elif resolve_method == "normfrac":
    #     sign_to_mult = normfrac_based_sign(Tensor)
    # elif resolve_method == "normmass":
    #     sign_to_mult = normmass_based_sign(Tensor)
    # else:
    #     raise ValueError(f"Sign resolve method {resolve_method} is not defined.")
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def topk_mask_preserve_normfrac(T, normfrac=0.9, return_mask=False):
    row_norms = torch.norm(T, p=2, dim=1, keepdim=True)

    # Calculate the proportion of each element's contribution to its row's norm
    proportion = T.abs() ** 2 / row_norms ** 2

    # Sort the proportions and their indices in descending order
    sorted_proportions, sorted_indices = torch.sort(proportion, dim=1, descending=True)

    # Calculate the cumulative sum of proportions
    cumsum_proportions = torch.cumsum(sorted_proportions, dim=1)

    # Find the indices where cumulative sum >= normfrac
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)

    # Create a range tensor to compare with normfrac_indices
    range_tensor = torch.arange(T.size(1)).unsqueeze(0).expand(T.size(0), -1)

    # Create a mask based on the normfrac_indices
    mask = range_tensor <= normfrac_indices.unsqueeze(1)

    # Initialize final_indices with a value that is out of bounds
    final_indices = torch.full_like(sorted_indices, T.size(1) - 1)

    # Use the mask to get the final indices
    final_indices[mask] = sorted_indices[mask]

    # Initialize the mask with zeros
    M = torch.zeros_like(T, dtype=torch.bool)

    # Use the final indices to update the final mask M
    M.scatter_(1, final_indices, True)

    if return_mask:
        return (T * M), M.float().mean(dim=1), M
    else:
        return (T * M), M.float().mean(dim=1)


def basic_merging(agg_mode, finetune_vectors, finetune_statedicts, remove_keys):
    # ft : the finetuned checkpoint state dict
    """ "Basic aggregation of the delta checks"""
    # probably not necessary
    #all_checks = flat_checks.clone()
    merged_vector = aggregate(finetune_vectors, agg_mode, final_signs=None)
    merged_statedict = vector_to_state_dict(merged_vector, finetune_statedicts[0], remove_keys=remove_keys)
    return merged_statedict


def task_vector_merging(task_vectors):
    """Merging by creating and scaling Task Vectors"""
    #all_checks = tv_flat_checks.clone()
    merged_task_vector = aggregate(task_vectors, "sum", final_signs=None)
    return merged_task_vector


def tie_merge(finetune_vectors, finetune_statedicts, pretrained_vector, remove_keys,reset,resolve,agg_mode,lamda):
    '''
    reset: the criterion for zero mask
    resolve: 
    agg_mode:
    lambda_code:
    '''

    task_vectors = finetune_vectors - pretrained_vector
    
    # trim
    if '_' not in reset:
        reset_type = ""
        reset_thresh = "none"
    else:
        reset_type,reset_thresh = reset.split('_')

    if reset_type=='nf':
        updated_task_vectors, *_ = topk_mask_preserve_normfrac(
            task_vectors, reset_thresh, return_mask=False
        )
    elif reset_type == "topk":
        updated_task_vectors = topk_values_mask(
            task_vectors, K=reset_thresh, return_mask=False
        )
    elif reset_type == "std": 
        raise NotImplementedError
        updated_task_vectors, *_ = greater_than_std_mask(
            task_vectors, reset_thresh, return_mask=False
        )
    else:
        #logger.info("Not removing NOISE")
        updated_task_vectors = task_vectors



    if resolve != "none":
        final_signs = resolve_sign(updated_task_vectors, resolve)
        assert final_signs is not None
    else:
        final_signs = None

    if "dis" in agg_mode:
        merged_task_vector = disjoint_aggregate(updated_task_vectors, agg_mode, final_signs)
    else:
        merged_task_vector = aggregate(updated_task_vectors, agg_mode, final_signs)

    
    # Delete to clean up CPU memory
    reference_state_dict = finetune_statedicts[0]
    merged_vector = pretrained_vector + lamda * merged_task_vector
    merged_statedict = vector_to_state_dict(
        merged_vector, reference_state_dict, remove_keys=remove_keys
    )

    return merged_statedict