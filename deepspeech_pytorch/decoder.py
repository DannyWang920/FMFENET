#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import torch
from six.moves import xrange
import kenlm

class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self,
                 labels,
                 lm_path=None,
                 alpha=0,
                 beta=0,
                 cutoff_top_n=40,
                 cutoff_prob=1.0,
                 beam_width=100,
                 num_processes=4,
                 blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        labels = list(labels)  # Ensure labels are a list before passing to decoder
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence,
                       size,
                       remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets


class BeamSearchDecoder(Decoder):
    def __init__(self, labels, beam_width=10, blank_index=0):
        super(BeamSearchDecoder, self).__init__(labels, blank_index)
        self.beam_width = beam_width

    def decode(self, probs, sizes=None):
        batch_size = probs.size(0)
        decoded_outputs = []
        for b in range(batch_size):
            beams = [([], 0)]  # (sequence, score)
            for t in range(sizes[b]):
                new_beams = []
                for seq, score in beams:
                    for i in range(probs.size(-1)):
                        new_seq = seq + [i]
                        new_score = score + probs[b, t, i].item()
                        new_beams.append((new_seq, new_score))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_width]
            best_seq = beams[0][0]
            decoded_outputs.append(self.convert_to_strings([best_seq], sizes=[len(best_seq)])[0])
        return decoded_outputs, None

    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence,
                       size,
                       remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            #char = self.int_to_char[sequence[i].item()]
            index = sequence[i].item() if isinstance(sequence[i], torch.Tensor) else sequence[i]
            char = self.int_to_char[index]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)


class BeamSearchDecoder(Decoder):
    def __init__(self, labels, blank_index=0, beam_width=3):
        super(BeamSearchDecoder, self).__init__(labels, blank_index)
        self.beam_width = beam_width  # 设置 beam width

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """将数值序列转换为字符串"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            # 如果 sequence[i] 是张量，使用 .item() 获取其值，否则直接使用 int
            char_index = sequence[i].item() if isinstance(sequence[i], torch.Tensor) else sequence[i]
            char = self.int_to_char[char_index]
            
            if char != self.int_to_char[self.blank_index]:
                # 如果 remove_repetitions=true，跳过重复字符
                if remove_repetitions and i != 0:
                    prev_char_index = sequence[i - 1].item() if isinstance(sequence[i - 1], torch.Tensor) else sequence[i - 1]
                    prev_char = self.int_to_char[prev_char_index]
                    if char == prev_char:
                        continue
                if char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string += char
                    offsets.append(i)
                    
        return string, torch.tensor(offsets, dtype=torch.int)


    def decode(self, probs, sizes=None):
        """
        使用 Beam Search 进行解码，返回最佳路径
        """
        batch_size, seq_length, num_classes = probs.shape
        beam_results = []

        for b in range(batch_size):
            sequences = [(list(), 0.0)]  # 初始序列为空，得分为0
            for t in range(seq_length):
                all_candidates = list()
                for seq, score in sequences:
                    for c in range(num_classes):
                        new_seq = seq + [c]
                        new_score = score - torch.log(probs[b, t, c])  # 使用负对数概率作为损失

                        all_candidates.append((new_seq, new_score))

                # 按照得分排序，保留前 beam_width 个最优序列
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                sequences = ordered[:self.beam_width]

            # 将最佳序列（得分最低）添加到结果中
            beam_results.append(sequences[0][0])

        strings, offsets = self.convert_to_strings(beam_results,
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets
