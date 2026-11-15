import torch

def is_tie_word_embeddings(model):
    if not model.config.tie_word_embeddings:
        print('The model does not enable tie_word_embeddings')
        return False
    state_dict = model.state_dict()
    source_param = state_dict['model.embed_tokens.weight']
    target_param = state_dict['lm_head.weight']
    if not source_param.view(-1)[1] == target_param.view(-1)[1]:
        print('The model has a wrong implementation for tie_word_embeddings')
        return False
    orig_value = source_param.view(-1)[1]
    new_value = 11.22
    source_param.view(-1)[1] = new_value
    if not source_param.view(-1)[1] == target_param.view(-1)[1]:
        print('embed_tokens and lm_head are not sharing the weight')
        return False
    if not target_param.view(-1)[1] == new_value:
        print('embed_tokens and lm_head are not sharing the weight')
        return False
    source_param.view(-1)[1] = orig_value
    return True