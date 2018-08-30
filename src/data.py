import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset

# for debug
from IPython import embed


class Vocab(object):

    def __init__(self, vocab_dict, add_pad, add_unk):
        self._vocab_dict = vocab_dict.copy()
        self._reverse_vocab_dict = dict()
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = len(self._vocab_dict)
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = len(self._vocab_dict)
            self._vocab_dict[self.unk_word] = self.unk_id
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    @classmethod
    def from_file(cls, path, add_pad=True, add_unk=True, max_size=None):
        vocab_dict = dict()
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_size and i >= max_size:
                    break
                word = line.strip().split()[0]
                vocab_dict[word] = len(vocab_dict)
        return cls(vocab_dict=vocab_dict, add_pad=add_pad, add_unk=add_unk)

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return len(self._vocab_dict)


class SentenceDataset(Dataset):

    def __init__(self, word_vocab, data_path, label_vocab, max_length, lower, min_length=2):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lower = lower
        self._max_length = max_length
        self._min_length = min_length
        self._data = []
        failed_to_parse = 0
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                try:
                    converted = self._convert_obj(obj)
                    if converted:
                        self._data.append(converted)
                    else:
                        failed_to_parse += 1
                except ValueError:
                    failed_to_parse += 1
                except AttributeError:
                    failed_to_parse += 1
        print('Failed to parse {:d} instances'.format(failed_to_parse))

    def _convert_obj(self, obj):
        sentence = obj['sentence']
        if self.lower:
            sentence = sentence.lower()
        word_ids = [self.word_vocab.word_to_id(w) for w in sentence.split()]
        length = len(word_ids)
        if length > self._max_length or length < self._min_length:
            return None
        label = self.label_vocab.word_to_id(obj['label'])
        if 'constituency_tree_encoding' in obj:
            mask_ids = self.tree_encoding_to_mask_ids(obj['constituency_tree_encoding'])
        else:
            mask_ids = None
        if mask_ids is not None and len(mask_ids) != length - 1:
            return None
        return word_ids, sentence, length, mask_ids, label

    def pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    @staticmethod
    def pad_mask(data):
        max_length = max(len(d) for d in data)
        padded = [d + [0] * (max_length - len(d)) for d in data]
        return np.array(padded)

    @staticmethod
    def make_one_hot_gold_mask(gold_mask_ids):
        masks = list()
        for num_classes in range(gold_mask_ids.shape[1], 1, -1):
            i = gold_mask_ids.shape[1] - num_classes
            indices = torch.LongTensor(gold_mask_ids[:, i])
            mask = SentenceDataset.convert_to_one_hot(indices, num_classes).float()
            masks.append(mask)
        return masks

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        words_batch, raw_sentences_batch, length_batch, mask_ids_batch, label_batch = list(zip(*batch))
        sentences = torch.LongTensor(self.pad_sentence(words_batch))
        lengths = torch.LongTensor(length_batch)
        try:
            masks = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_batch))
        except TypeError:
            masks = None
        labels = torch.LongTensor(label_batch)
        return {'sentences': sentences, 'lengths': lengths, 'masks': masks, 'labels': labels,
                'raw_sentences': raw_sentences_batch}

    @staticmethod
    def tree_encoding_to_mask_ids(tree_encoding):
        items = [int(x) for x in tree_encoding.strip().split(',')]
        sentence_length = len(items) // 3 + 1
        curr_sent = [x for x in range(sentence_length)]
        mask_ids = list()
        assert 3 * (sentence_length - 1) == len(items)
        for i in range(sentence_length - 1):
            left_node = items[i * 3]
            right_node = items[i * 3 + 1]
            father_node = items[i * 3 + 2]
            left_index = curr_sent.index(left_node)
            assert curr_sent[left_index + 1] == right_node
            curr_sent = curr_sent[:left_index] + [father_node] + curr_sent[left_index + 2:]
            mask_ids.append(left_index)
        return mask_ids

    @staticmethod
    def convert_to_one_hot(indices, num_classes):
        batch_size = indices.shape[0]
        indices = indices.unsqueeze(1)
        one_hot = indices.new(batch_size, num_classes).zero_().scatter_(1, indices, 1)
        return one_hot


class QuickThoughtDataset(Dataset):
    def __init__(self, word_vocab, data_path, max_length, lower):
        self.word_vocab = word_vocab
        self.lower = lower
        self._max_length = max_length
        self._data = []
        failed_to_parse = 0
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                try:
                    converted = self._convert_obj(obj)
                    if converted:
                        self._data.append(converted)
                    else:
                        failed_to_parse += 1
                except ValueError:
                    failed_to_parse += 1
                except AttributeError:
                    failed_to_parse += 1
        print('Failed to parse {:d} instances'.format(failed_to_parse))

    def _convert_obj(self, obj):
        sentence = obj['sentence']
        context = obj['context']
        if self.lower:
            sentence = sentence.lower()
            context = context.lower()
        word_ids_s = [self.word_vocab.word_to_id(w) for w in sentence.split()]
        length_s = len(word_ids_s)
        word_ids_c = [self.word_vocab.word_to_id(w) for w in context.split()]
        length_c = len(word_ids_c)
        if length_s > self._max_length or length_s < 2 or length_c > self._max_length or length_c < 2:
            return None
        if 'constituency_tree_encoding_s' in obj:
            mask_ids_s = self.tree_encoding_to_mask_ids(obj['constituency_tree_encoding_s'])
            mask_ids_c = self.tree_encoding_to_mask_ids(obj['constituency_tree_encoding_c'])
        else:
            mask_ids_s = mask_ids_c = None
        if mask_ids_s is not None and len(mask_ids_s) != length_s - 1:
            return None
        if mask_ids_c is not None and len(mask_ids_c) != length_c - 1:
            return None
        return word_ids_s, word_ids_c, sentence, context, length_s, length_c, mask_ids_s, mask_ids_c

    def pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    @staticmethod
    def pad_mask(data):
        max_length = max(len(d) for d in data)
        padded = [d + [0] * (max_length - len(d)) for d in data]
        return np.array(padded)

    @staticmethod
    def make_one_hot_gold_mask(gold_mask_ids):
        masks = list()
        for num_classes in range(gold_mask_ids.shape[1], 1, -1):
            i = gold_mask_ids.shape[1] - num_classes
            indices = torch.LongTensor(gold_mask_ids[:, i])
            mask = SentenceDataset.convert_to_one_hot(indices, num_classes).float()
            masks.append(mask)
        return masks

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        words_s, words_c, raw_sentences_s, raw_sentences_c, lengths_s, lengths_c, mask_ids_s, mask_ids_c = list(
            zip(*batch))
        sentences_s = torch.LongTensor(self.pad_sentence(words_s))
        sentences_c = torch.LongTensor(self.pad_sentence(words_c))
        lengths_s = torch.LongTensor(lengths_s)
        lengths_c = torch.LongTensor(lengths_c)
        try:
            masks_s = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_s))
            masks_c = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_c))
        except TypeError:
            masks_s = masks_c = None
        return {'sentences_s': sentences_s,
                'sentences_c': sentences_c,
                'raw_sentences_s': raw_sentences_s,
                'raw_sentences_c': raw_sentences_c,
                'lengths_s': lengths_s,
                'lengths_c': lengths_c,
                'masks_s': masks_s,
                'masks_c': masks_c}

    @staticmethod
    def tree_encoding_to_mask_ids(tree_encoding):
        items = [int(x) for x in tree_encoding.strip().split(',')]
        sentence_length = len(items) // 3 + 1
        curr_sent = [x for x in range(sentence_length)]
        mask_ids = list()
        assert 3 * (sentence_length - 1) == len(items)
        for i in range(sentence_length - 1):
            left_node = items[i * 3]
            right_node = items[i * 3 + 1]
            father_node = items[i * 3 + 2]
            left_index = curr_sent.index(left_node)
            assert curr_sent[left_index + 1] == right_node
            curr_sent = curr_sent[:left_index] + [father_node] + curr_sent[left_index + 2:]
            mask_ids.append(left_index)
        return mask_ids

    @staticmethod
    def convert_to_one_hot(indices, num_classes):
        batch_size = indices.shape[0]
        indices = indices.unsqueeze(1)
        one_hot = indices.new(batch_size, num_classes).zero_().scatter_(1, indices, 1)
        return one_hot


class NLIDataset(Dataset):
    def __init__(self, word_vocab, data_path, max_length, lower):
        self.word_vocab = word_vocab
        self.lower = lower
        self._max_length = max_length
        self._data = []
        failed_to_parse = 0
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                try:
                    converted = self._convert_obj(obj)
                    if converted:
                        self._data.append(converted)
                    else:
                        failed_to_parse += 1
                except ValueError:
                    failed_to_parse += 1
                except AttributeError:
                    failed_to_parse += 1
        print('Failed to parse {:d} instances'.format(failed_to_parse))

    def _convert_obj(self, obj):
        sentence1 = obj['sentence1']
        sentence2 = obj['sentence2']
        if self.lower:
            sentence1 = sentence1.lower()
            sentence2 = sentence2.lower()
        word_ids_1 = [self.word_vocab.word_to_id(w) for w in sentence1.split()]
        length_1 = len(word_ids_1)
        word_ids_2 = [self.word_vocab.word_to_id(w) for w in sentence2.split()]
        length_2 = len(word_ids_2)
        label = obj['gold_label']
        if length_1 > self._max_length or length_1 < 2 or length_2 > self._max_length or length_2 < 2:
            return None
        if 'sentence1_binary_encoding' in obj:
            mask_ids_1 = self.tree_encoding_to_mask_ids(obj['sentence1_binary_encoding'])
            mask_ids_2 = self.tree_encoding_to_mask_ids(obj['sentence2_binary_encoding'])
        else:
            mask_ids_1 = mask_ids_2 = None
        if mask_ids_1 is not None and len(mask_ids_1) != length_1 - 1:
            return None
        if mask_ids_2 is not None and len(mask_ids_2) != length_2 - 1:
            return None
        return word_ids_1, word_ids_2, sentence1, sentence2, length_1, length_2, mask_ids_1, mask_ids_2, label

    def pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    @staticmethod
    def pad_mask(data):
        max_length = max(len(d) for d in data)
        padded = [d + [0] * (max_length - len(d)) for d in data]
        return np.array(padded)

    @staticmethod
    def make_one_hot_gold_mask(gold_mask_ids):
        masks = list()
        for num_classes in range(gold_mask_ids.shape[1], 1, -1):
            i = gold_mask_ids.shape[1] - num_classes
            indices = torch.LongTensor(gold_mask_ids[:, i])
            mask = SentenceDataset.convert_to_one_hot(indices, num_classes).float()
            masks.append(mask)
        return masks

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        words_1, words_2, raw_sentences_1, raw_sentences_2, \
            lengths_1, lengths_2, mask_ids_1, mask_ids_2, labels = list(
                zip(*batch))
        sentences_1 = torch.LongTensor(self.pad_sentence(words_1))
        sentences_2 = torch.LongTensor(self.pad_sentence(words_2))
        lengths_1 = torch.LongTensor(lengths_1)
        lengths_2 = torch.LongTensor(lengths_2)
        labels = torch.LongTensor(labels)
        try:
            masks_1 = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_1))
            masks_2 = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_2))
        except TypeError:
            masks_1 = masks_2 = None
        return {'sentences_1': sentences_1,
                'sentences_2': sentences_2,
                'raw_sentences_1': raw_sentences_1,
                'raw_sentences_2': raw_sentences_2,
                'lengths_1': lengths_1,
                'lengths_2': lengths_2,
                'masks_1': masks_1,
                'masks_2': masks_2,
                'labels': labels}

    @staticmethod
    def tree_encoding_to_mask_ids(tree_encoding):
        items = [int(x) for x in tree_encoding.strip().split(',')]
        sentence_length = len(items) // 3 + 1
        curr_sent = [x for x in range(sentence_length)]
        mask_ids = list()
        assert 3 * (sentence_length - 1) == len(items)
        for i in range(sentence_length - 1):
            left_node = items[i * 3]
            right_node = items[i * 3 + 1]
            father_node = items[i * 3 + 2]
            left_index = curr_sent.index(left_node)
            assert curr_sent[left_index + 1] == right_node
            curr_sent = curr_sent[:left_index] + [father_node] + curr_sent[left_index + 2:]
            mask_ids.append(left_index)
        return mask_ids

    @staticmethod
    def convert_to_one_hot(indices, num_classes):
        batch_size = indices.shape[0]
        indices = indices.unsqueeze(1)
        one_hot = indices.new(batch_size, num_classes).zero_().scatter_(1, indices, 1)
        return one_hot


class TranslationDataset(Dataset):

    def __init__(self, src_word_vocab, tgt_word_vocab, data_path, max_length, lower, min_length=2):
        self.src_word_vocab = src_word_vocab
        self.tgt_word_vocab = tgt_word_vocab
        self.lower = lower
        self._max_length = max_length
        self._min_length = min_length
        self._data = []
        failed_to_parse = 0
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                try:
                    converted = self._convert_obj(obj)
                    if converted:
                        self._data.append(converted)
                    else:
                        failed_to_parse += 1
                except ValueError:
                    failed_to_parse += 1
                except AttributeError:
                    failed_to_parse += 1
        print('Failed to parse {:d} instances'.format(failed_to_parse))

    def _convert_obj(self, obj):
        source = obj['source_sentence']
        target = obj['target_sentence']
        if self.lower:
            source = source.lower()
        src_word_ids = [self.src_word_vocab.word_to_id(w) for w in source.split()]
        tgt_word_ids = [self.tgt_word_vocab.word_to_id(w) for w in target.split()]
        src_length = len(src_word_ids)
        tgt_length = len(tgt_word_ids)
        if src_length > self._max_length or src_length < self._min_length or tgt_length > self._max_length:
            return None
        if 'parsed_source_sentence' in obj:
            mask_ids = self.tree_encoding_to_mask_ids(obj['parsed_source_sentence'])
        else:
            mask_ids = None
        if mask_ids is not None and len(mask_ids) != src_length - 1:
            return None
        return src_word_ids, tgt_word_ids, source, target, src_length, tgt_length, mask_ids

    @staticmethod
    def pad_sentence(data, vocab):
        max_length = max(len(d) for d in data)
        padded = [d + [vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    @staticmethod
    def pad_mask(data):
        max_length = max(len(d) for d in data)
        padded = [d + [0] * (max_length - len(d)) for d in data]
        return np.array(padded)

    @staticmethod
    def make_one_hot_gold_mask(gold_mask_ids):
        masks = list()
        for num_classes in range(gold_mask_ids.shape[1], 1, -1):
            i = gold_mask_ids.shape[1] - num_classes
            indices = torch.LongTensor(gold_mask_ids[:, i])
            mask = SentenceDataset.convert_to_one_hot(indices, num_classes).float()
            masks.append(mask)
        return masks

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        src_word_batch, tgt_word_batch, source_sentence_batch, target_sentence_batch, \
            src_lengths_batch, tgt_lengths_batch, mask_ids_batch = list(zip(*batch))
        tgt_input_word_batch = [vec[:-1] for vec in tgt_word_batch]
        tgt_output_word_batch = [vec[1:] for vec in tgt_word_batch]
        src_words = torch.LongTensor(self.pad_sentence(src_word_batch, self.src_word_vocab))
        tgt_input_words = torch.LongTensor(self.pad_sentence(tgt_input_word_batch, self.tgt_word_vocab))
        tgt_output_words = torch.LongTensor(self.pad_sentence(tgt_output_word_batch, self.tgt_word_vocab))
        src_lengths = torch.LongTensor(src_lengths_batch)
        tgt_lengths = torch.LongTensor(tgt_lengths_batch)
        try:
            masks = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_batch))
        except TypeError:
            masks = None
        return {'sources': src_words, 'targets_input': tgt_input_words, 'targets_output': tgt_output_words,
                'masks': masks, 'src_lengths': src_lengths, 'tgt_lengths': tgt_lengths,
                'src_sentences': source_sentence_batch, 'tgt_sentences': target_sentence_batch}

    @staticmethod
    def tree_encoding_to_mask_ids(tree_encoding):
        items = [int(x) for x in tree_encoding.strip().split(',')]
        sentence_length = len(items) // 3 + 1
        curr_sent = [x for x in range(sentence_length)]
        mask_ids = list()
        assert 3 * (sentence_length - 1) == len(items)
        for i in range(sentence_length - 1):
            left_node = items[i * 3]
            right_node = items[i * 3 + 1]
            father_node = items[i * 3 + 2]
            left_index = curr_sent.index(left_node)
            assert curr_sent[left_index + 1] == right_node
            curr_sent = curr_sent[:left_index] + [father_node] + curr_sent[left_index + 2:]
            mask_ids.append(left_index)
        return mask_ids

    @staticmethod
    def convert_to_one_hot(indices, num_classes):
        batch_size = indices.shape[0]
        indices = indices.unsqueeze(1)
        one_hot = indices.new(batch_size, num_classes).zero_().scatter_(1, indices, 1)
        return one_hot
