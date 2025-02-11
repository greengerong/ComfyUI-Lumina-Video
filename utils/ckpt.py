def remove_wrapper_name_from_state_dict(state_dict):
    new_ckpt = {}
    for key, val in state_dict.items():
        new_ckpt[key.replace("_orig_mod.", "")] = val

    return new_ckpt
