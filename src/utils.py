import nltk
import numpy as np
import torch
from torch.autograd import Variable

from src.data import SentenceDataset

from IPython import embed


def bleu(hypothesis, reference, eos_token):
    hypothesis = hypothesis.data.tolist()
    reference = reference.data.tolist()
    for i in range(len(hypothesis)):
        if hypothesis[i] == eos_token:
            hypothesis = hypothesis[:i+1]
            break
    for i in range(len(reference)):
        if reference[i] == eos_token:
            reference = reference[:i+1]
            break
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis) * 100


def wrap_with_variable(tensor, volatile=False):
    var = Variable(tensor, volatile=volatile)
    if torch.cuda.is_available():
        var = var.cuda()
    return var


def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var


def get_tree_structures(sents, lengths, masks):
    tree_strs = list()
    for sent_id, sent in enumerate(sents):
        words = sent.split()
        for word_id in range(unwrap_scalar_variable(lengths[sent_id]) - 2):
            selected_id = unwrap_scalar_variable(masks[word_id][sent_id].max(0)[1])
            words = words[:selected_id] + ['( {:s} {:s} )'.format(
                words[selected_id], words[selected_id+1])] + words[selected_id+2:]
        assert len(words) == 2
        tree_strs.append('( {:s} {:s} )'.format(words[0], words[1]))
    return tree_strs


def build_balance_tree_from_range(left, right, curr_node_idx, node_map=None):
    if left == right:
        if not node_map:
            return None, left, curr_node_idx
        else:
            return None, node_map[left], curr_node_idx
    mid = (left + right) // 2
    left_str, left_root, curr_node_idx = build_balance_tree_from_range(left, mid, curr_node_idx, node_map)
    right_str, right_root, curr_node_idx = build_balance_tree_from_range(mid+1, right, curr_node_idx, node_map)
    if (left_str is not None) and (right_str is not None):
        child_spans_str = left_str + ',' + right_str
    elif (left_str is not None) and (right_str is None):
        child_spans_str = left_str
    elif (left_str is None) and (right_str is not None):
        child_spans_str = right_str
    else:
        child_spans_str = None
    curr_span_str = str(left_root) + ',' + str(right_root) + ',' + str(curr_node_idx)
    curr_node_idx += 1
    if child_spans_str is not None:
        tree_str = child_spans_str + ',' + curr_span_str
    else:
        tree_str = curr_span_str
    return tree_str, curr_node_idx - 1, curr_node_idx


def generate_balance_masks(lengths):
    tree_mask_ids = list()
    for length in lengths:
        tree_str, _, curr_node_idx = build_balance_tree_from_range(0, length-1, length)
        assert curr_node_idx == 2 * length - 1
        tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(tree_str))
    masks = SentenceDataset.make_one_hot_gold_mask(SentenceDataset.pad_mask(tree_mask_ids))
    for i, mask in enumerate(masks):
        masks[i] = wrap_with_variable(mask, False)
    return masks


def generate_guided_balance_masks(lengths, query_pos):
    tree_mask_ids = list()
    for j, length in enumerate(lengths):
        left_range = max(query_pos[j] - 4, 0)
        right_range = min(query_pos[j] + 4, length - 1)
        subtree_str, subtree_root, curr_node_idx = build_balance_tree_from_range(
            left_range, right_range, length)
        node_map = list()
        for i in range(left_range):
            node_map.append(i)
        node_map.append(subtree_root)
        for i in range(right_range + 1, length):
            node_map.append(i)
        if len(node_map) == 1:
            tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(subtree_str))
            continue
        full_tree_str, _, curr_node_idx = build_balance_tree_from_range(
            0, len(node_map) - 1, curr_node_idx, node_map)
        tree_str = subtree_str + ',' + full_tree_str
        assert curr_node_idx == 2 * length - 1
        tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(tree_str))
    masks = SentenceDataset.make_one_hot_gold_mask(SentenceDataset.pad_mask(tree_mask_ids))
    for i, mask in enumerate(masks):
        masks[i] = wrap_with_variable(mask, False)
    return masks


def generate_lifted_balance_masks(lengths, query_pos):
    tree_mask_ids = list()
    for j, length in enumerate(lengths):
        left_range = max(query_pos[j] - 4, 0)
        right_range = min(query_pos[j] + 4, length - 1)
        subtree_str, subtree_root, curr_node_idx = build_balance_tree_from_range(
            left_range, right_range, length)
        if left_range > 1:
            left_tree_str, left_tree_root, curr_node_idx = build_balance_tree_from_range(
                0, left_range - 1, curr_node_idx
            )
        elif left_range == 1:
            left_tree_str = None
            left_tree_root = 0
        else:
            left_tree_str = left_tree_root = None
        node_map = list()
        for i in range(right_range + 1, length):
            node_map.append(i)
        if len(node_map) > 1:
            right_tree_str, right_tree_root, curr_node_idx = build_balance_tree_from_range(
                right_range + 1, length - 1, curr_node_idx
            )
        elif len(node_map) == 1:
            right_tree_str = None
            right_tree_root = node_map[0]
        else:
            right_tree_str = right_tree_root = None
        node_map_candidate = list(
            filter(lambda x: x[1] is not None, [(left_tree_str, left_tree_root),
                                                (subtree_str, subtree_root),
                                                (right_tree_str, right_tree_root)]))
        if len(node_map_candidate) == 1:
            tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(subtree_str))
            continue
        node_map = [x[1] for x in node_map_candidate]
        full_tree_str, _, curr_node_idx = build_balance_tree_from_range(
            0, len(node_map) - 1, curr_node_idx, node_map)
        tree_str = subtree_str
        if left_tree_str is not None:
            tree_str += ',' + left_tree_str
        if right_tree_str is not None:
            tree_str += ',' + right_tree_str
        tree_str += ',' + full_tree_str
        assert curr_node_idx == 2 * length - 1
        tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(tree_str))
    masks = SentenceDataset.make_one_hot_gold_mask(SentenceDataset.pad_mask(tree_mask_ids))
    for i, mask in enumerate(masks):
        masks[i] = wrap_with_variable(mask, False)
    return masks


def build_left_branch_tree(length):
    if length == 1:
        return ''
    tree_str = '0,1,{:d}'.format(length)
    for i in range(2, length):
        tree_str += ',{:d},{:d},{:d}'.format(length+i-2, i, length+i-1)
    return tree_str


def generate_left_branch_masks(lengths):
    tree_mask_ids = list()
    for length in lengths:
        tree_str = build_left_branch_tree(length)
        tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(tree_str))
    masks = SentenceDataset.make_one_hot_gold_mask(SentenceDataset.pad_mask(tree_mask_ids))
    for i, mask in enumerate(masks):
        masks[i] = wrap_with_variable(mask, False)
    return masks


def build_right_branch_tree(length):
    if length == 1:
        return ''
    tree_str = '{:d},{:d},{:d}'.format(length-2, length-1, length)
    curr_node = length
    for i in range(3, length + 1):
        tree_str += ',{:d},{:d},{:d}'.format(length - i, curr_node, curr_node + 1)
        curr_node += 1
    return tree_str


def generate_right_branch_masks(lengths):
    tree_mask_ids = list()
    for length in lengths:
        tree_str = build_right_branch_tree(length)
        tree_mask_ids.append(SentenceDataset.tree_encoding_to_mask_ids(tree_str))
    masks = SentenceDataset.make_one_hot_gold_mask(SentenceDataset.pad_mask(tree_mask_ids))
    for i, mask in enumerate(masks):
        masks[i] = wrap_with_variable(mask, False)
    return masks


def load_glove(path, vocab, init_weight):
    word_vectors = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, *values = line.split()
            try:
                if vocab.has_word(word):
                    if word in word_vectors:
                        # Let's use the first occurrence only.
                        continue
                    word_vector = np.array([float(v) for v in values])
                    word_vectors[word] = word_vector
            except ValueError:
                # 840D GloVe file has some encoding errors...
                # I think they can be ignored.
                continue
    glove_weight = np.zeros_like(init_weight)
    # glove_weight[:] = word_vectors[vocab.unk_word]
    for word in word_vectors:
        word_index = vocab.word_to_id(word)
        glove_weight[word_index, :] = word_vectors[word]
    return glove_weight


def sort_sentences_by_lengths(x, lengths, code=None):
    lengths, idx = lengths.sort(0, descending=True)
    x = x.index_select(0, idx)
    _, inv = idx.sort(0)
    if code is not None:
        code = code.index_select(0, idx)
        return x, lengths, inv, code
    else:
        return x, lengths, inv
