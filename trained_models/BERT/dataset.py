# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT datasets."""

__all__ = ['DatasetWrapper', 'ClassificationTransform', 'BERTTransform']

import io
import os
import numpy as np
try:
    from tokenizer import convert_to_unicode
except ImportError:
    from .tokenizer import convert_to_unicode
from gluonnlp.data import TSVDataset
from gluonnlp.data.registry import register

def line_splitter(s):
    return s.split('\n')

class DatasetWrapper(TSVDataset):
    """ Simple wrapper around TSVDataset. """
    def __init__(self, path, field_separator = None):
        super(DatasetWrapper, self).__init__(path, 
                field_separator = field_separator, 
                sample_splitter = line_splitter)
    
    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1', '2', '3']
    
    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding) as fin:
                content = fin.read()
            raw_samples = self._sample_splitter(content)
            samples_data = list(filter(lambda x: len(x) > 0, raw_samples))
            samples = (s for s in samples_data if not self._should_discard())
            if self._field_separator:
                samples = [self._field_selector(self._field_separator(s)) for s in samples]
            all_samples += samples
        return all_samples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class BERTTransform(object):
    """BERT style data transformation.
    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """
    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.
        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length
        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.
        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
        For single sequences, the input is a tuple of single string: text_a.
        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).
        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        """
        # convert to unicode
        text_a = line[0]
        text_a = convert_to_unicode(text_a)
        if self._pair:
            assert len(line) == 2
            text_b = line[1]
            text_b = convert_to_unicode(text_b)

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            input_ids.extend([0] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
               np.array(segment_ids, dtype='int32')

class ClassificationTransform(object):
    """Dataset Transformation for BERT-style Sentence Classification.
    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    labels : list of int.
        List of all label ids for the classification task.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """
    def __init__(self, tokenizer, labels, max_seq_length, pad=True, pair=True):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._bert_xform = BERTTransform(tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.
        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length
        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.
        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0
        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1
        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).
        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: label id in 'int32', shape (batch_size, 1)
        """
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')
        input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
        return input_ids, valid_length, segment_ids, label_id
