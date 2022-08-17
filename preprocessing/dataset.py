from tqdm import tqdm
import random
import ast
import collections
import torch
import os
import unicodedata
from io import open
from random import sample
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
import sentencepiece as spm
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Use bos to represent MASK
MASK='<MASK>'
lacode = {"<wu>":"<wuu>", "<yu>":"<yue>", "<zs>":"<zsm>"}

"""
french_letters = ['é',
        'à', 'è', 'ù',
        'â', 'ê', 'î', 'ô', 'û',
        'ç',
        'ë', 'ï', 'ü']
"""
commas = ['.','?','!',',','"',':',';','<','>','-','+','=',' ','_', "'",'%','&','#','@','*','(',')','~','`']
"""
nums = list('0123456789')
allowed = list('abcdefghijklmnopqrstuvwxyz') + french_letters + commas + nums
"""

def remove_other_letters(text):
    """
    if '_' + la1 == text[-3:] or '_' + la2 == text[-3:]:
        return text[:-3]
    elif '<<split>>' == text:
        return '<<split>>'
    """
    if '<<split>>' == text:
        return '<<split>>'
    elif text[-3] == '_':
        return text[:-3]
    return ''


class RawInput(object):
    """A single training/test example for the language model.
        This example should contains many sentences in different languages
    """    
    def __init__(self, guid, sent1, sent2, la1, la2):
        """Constructs an InputExample.
        Args:
            guid: Unique id for the example.
            sent1: list of strings, represents English sequence. Each entry is untokenized text of individual sentences. For single
            sequence tasks
            sent2: list of strings, represents second French sequence.
        """
        self.guid = guid
        self.sent1 = sent1  # tokens in first language
        self.sent2 = sent2  # tokens in second language
        if la1 in lacode.keys():
            la1 = lacode[la1]
        if la2 in lacode.keys():
            la2 = lacode[la2]
        self.la1 = la1
        self.la2 = la2


def load_inputs(raw_file, config, data_type='train'):    
    """Loads raw data into a dictionary and return vocab and raw samples."""
    vocab = collections.OrderedDict()
    raw_inputs = []
    weights = []
    index = 0
    anomaly = 0
    
    if config.do_oversampling:
        with open(config.weights_path, "r") as f:
            data = f.read()
            weights_dic = ast.literal_eval(data)
        del weights_dic["en"]
        total_num = sum(weights_dic.values())
        for la in weights_dic.keys():
            weights_dic[la] = (weights_dic[la] / total_num) ** (1.0 / config.oversampling_T) / weights_dic[la] * 1e8
        
    with open(raw_file, "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading {} Dataset".format(data_type)):
            words = line.strip().lower().split(' ')
            la1 = '<' + words[0][-2:] + '>'
            la2 = '<' + words[-1][-2:] + '>'
            # processed = list(map(lambda word: remove_other_letters(word, config.lg1, config.lg2), words))
            processed = list(map(lambda word: remove_other_letters(word), words))
            if len(processed) <= 2:
                logger.warn('invalid line: {}'.format(line))
                anomaly += 1
                continue
            sp = processed.index('<<split>>')
            if sp==0 or sp==-1 or sp==len(processed)-1:
                logger.warn('invalid line: {} '.format(line))
                anomaly += 1
                continue
            rawinput = RawInput(index, ' '.join(processed[:sp]), ' '.join(processed[sp+1:]), la1, la2)
            # processed.insert(sp+1, la2)
            # processed.insert(0, la1)
            index += 1
            raw_inputs.append(rawinput)
            if config.do_oversampling:
                weight_la = la1 if la2 == "<en>" else la2
                weight = weights_dic[weight_la[1:-1]]
                weights.append(weight)
        print("anomaly_num = ", anomaly)
    return raw_inputs, weights


class BPETokenizer(object):
    def __init__(self, bpe_path='bpe.model', max_len=120):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(bpe_path)
        self.vocab_size = len(self.sp)
        self.max_len = max_len
        logger.info("BPE path: {}".format(bpe_path))
        logger.info('We have loaded tokenizer with vocab_size:{} and max_len:{}.'
                    .format(self.vocab_size, self.max_len))
        
    #sp.EncodeAsIds("This is a test")
    def convert_sents_to_ids(self, string):
        """ converts a raw corpus into ids"""
        ids = self.sp.encode_as_ids(string)
        if len(ids) > self.max_len:
            #logger.warn('One sentence has {} tokens which is longer than max_len:{}. Cut them off.'
            #            .format(len(ids),self.max_len))
            ids = ids[:self.max_len]
        return ids
    
    # sp.EncodeAsPieces("This is a test")
    def convert_sents_to_tokens(self, string):
        """ converts a raw corpus into tokens"""
        tokens = self.sp.encode_as_pieces(string)
        if len(tokens)>self.max_len:
            logger.warn('One sentence has {} tokens which is longer than max_len:{}. Cut them off.'
                        .format(len(tokens),self.max_len))
            tokens = tokens[:self.max_len]
        return tokens
    
    # sp.DecodePieces(['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'])
    def convert_tokens_to_string(self, ls):
        """ converts a list of tokens into original string"""
        if not isinstance(ls, list) and isinstance(ls, str):
            ls = [ls]
        return self.sp.decode_pieces(ls)
    
    def convert_ids_to_string(self, ls):
        """ converts a list of tokens into original string"""
        if not isinstance(ls,list) and isinstance(ls, int):
            ls = [ls]
        return self.sp.decode_ids(ls)


class ValiDataset(Dataset):
    def __init__(self, vali_samples, corpus_path, config, bpe_path='bpe.model', max_len=120, provide_valid=True):
        self.tokenizer = BPETokenizer(bpe_path, max_len)
        self.vocab_size = len(self.tokenizer.sp)
        self.max_len = max_len
        if provide_valid:
            raw_inputs, _ = load_inputs(corpus_path, config, 'All')
            self.inputs = raw_inputs
        else:
            self.inputs = vali_samples
        self.num_samples = len(self.inputs)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, item):

        raw_input = self.inputs[item]
        # tokenize
        ids_1 = self.tokenizer.convert_sents_to_ids(raw_input.sent1)
        ids_2 = self.tokenizer.convert_sents_to_ids(raw_input.sent2)
        sent1_tgt_la_id = self.tokenizer.sp.PieceToId(raw_input.la2[0] + "2" + raw_input.la2[1:]) - 3
        sent2_tgt_la_id = self.tokenizer.sp.PieceToId(raw_input.la1[0] + "2" + raw_input.la1[1:]) - 3
        # """
        # mask tokens
        if random.random() > 0.5:
            # Mask English
            i = 0
            while True:
                i += 1
                mask_id = random.randint(0, len(ids_1) - 1)
                if self.tokenizer.convert_ids_to_string(ids_1[mask_id]) not in commas and i < 6:
                    break
                else:
                    break
            tgt_lg = 1
            mask_label = ids_1[mask_id]
            ids_1[mask_id] = self.tokenizer.sp.PieceToId(MASK)
        else:
            # Mask French
            i = 0
            while True:
                i += 1
                mask_id = random.randint(0, len(ids_2) - 1)
                if self.tokenizer.convert_ids_to_string(ids_2[mask_id]) not in commas and i < 6:
                    break
                else:
                    break
            tgt_lg = 2
            mask_label = ids_2[mask_id]
            ids_2[mask_id] = self.tokenizer.sp.PieceToId(MASK)
        """
        cur_tensors = (torch.tensor(ids_1),
                       torch.tensor(len(ids_1)),
                       torch.tensor(ids_2),
                       torch.tensor(len(ids_2)),
                       torch.tensor(mask_label),
                       torch.tensor(tgt_lg),
                       torch.tensor(mask_id),
                       torch.tensor(raw_input.guid),
                       torch.tensor(sent1_tgt_la_id),
                       torch.tensor(sent2_tgt_la_id))
        """
        cur_tensors = (torch.tensor(ids_1),
                       torch.tensor(len(ids_1)),
                       torch.tensor(ids_2),
                       torch.tensor(len(ids_2)),
                       None,
                       None,
                       None,
                       torch.tensor(raw_input.guid),
                       torch.tensor(sent1_tgt_la_id),
                       torch.tensor(sent2_tgt_la_id))
        # """
        return cur_tensors


class InputDataset(Dataset):
    def __init__(self, corpus_path, validation_size, config, bpe_path='bpe.model', max_len=120, make_valid=True, provide_valid=False):
        self.tokenizer = BPETokenizer(bpe_path, max_len)
        self.vocab_size = len(self.tokenizer.sp)
        self.max_len = max_len
        raw_inputs, weights = load_inputs(corpus_path, config, 'All')
        if make_valid and not provide_valid:
            validation_indice = sample(range(len(raw_inputs)), validation_size)
            self.inputs = np.delete(raw_inputs, validation_indice)
            self.weights = np.delete(weights, validation_indice)
            self.vali_inputs = np.array(raw_inputs)[validation_indice]
        else:
            self.weights = weights
            self.inputs = raw_inputs
        self.num_samples = len(self.inputs)
        logger.info("Intialized sampling weight list with the length of %d.", len(self.weights))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):

        raw_input = self.inputs[item]
        # tokenize
        ids_1 = self.tokenizer.convert_sents_to_ids(raw_input.sent1)
        ids_2 = self.tokenizer.convert_sents_to_ids(raw_input.sent2)
        sent1_tgt_la_id = self.tokenizer.sp.PieceToId(raw_input.la2[0] + "2" + raw_input.la2[1:]) - 3
        sent2_tgt_la_id = self.tokenizer.sp.PieceToId(raw_input.la1[0] + "2" + raw_input.la1[1:]) - 3
        
        if ids_1 == [] or ids_2 == []:
            return (None, None, None, None, None, None, None, None, None, None)
        
        # """
        # mask tokens
        if random.random() > 0.5:
            # Mask English
            i = 0
            while True:
                i += 1
                mask_id = random.randint(0, len(ids_1)-1)
                if self.tokenizer.convert_ids_to_string(ids_1[mask_id]) not in commas and i < 6:
                    break
                else:
                    break
            tgt_lg = 1
            mask_label = ids_1[mask_id]
            ids_1[mask_id] = self.tokenizer.sp.PieceToId(MASK)
        else:
            # Mask French
            i = 0
            while True:
                i += 1
                mask_id = random.randint(0, len(ids_2)-1)
                if self.tokenizer.convert_ids_to_string(ids_2[mask_id]) not in commas and i < 6:
                    break
                else:
                    break
            tgt_lg = 2
            mask_label = ids_2[mask_id]
            ids_2[mask_id] = self.tokenizer.sp.PieceToId(MASK)
        """
        cur_tensors = (torch.tensor(ids_1),
                       torch.tensor(len(ids_1)),
                       torch.tensor(ids_2),
                       torch.tensor(len(ids_2)),
                       torch.tensor(mask_label),
                       torch.tensor(tgt_lg),
                       torch.tensor(mask_id),
                       torch.tensor(raw_input.guid),
                       torch.tensor(sent1_tgt_la_id),
                       torch.tensor(sent2_tgt_la_id))
        """

        cur_tensors = (torch.tensor(ids_1),
                       torch.tensor(len(ids_1)),
                       torch.tensor(ids_2),
                       torch.tensor(len(ids_2)),
                       None,
                       None,
                       None,
                       torch.tensor(raw_input.guid),
                       torch.tensor(sent1_tgt_la_id),
                       torch.tensor(sent2_tgt_la_id))            
        # """               
        return cur_tensors
    

def pad_seq(seq, length):
    if seq is None:
        return None
    return F.pad(seq, pad=(0, length - seq.shape[-1]), mode='constant', value=0)


def collate_fn(data):
    max_len1 = max(list(filter(lambda x: x is not None, [i[1] for i in data])))
    max_len2 = max(list(filter(lambda x: x is not None, [i[3] for i in data])))
    sent1 = torch.stack(list(filter(lambda x: x is not None, [pad_seq(i[0], max_len1) for i in data])))
    sent2 = torch.stack(list(filter(lambda x: x is not None, [pad_seq(i[2], max_len2) for i in data])))
    sent1_len = torch.stack(list(filter(lambda x: x is not None, [i[1] for i in data])))
    sent2_len = torch.stack(list(filter(lambda x: x is not None, [i[3] for i in data])))
    # lm_label = torch.stack(list(filter(lambda x: x is not None, [i[4] for i in data]))) # remove for XTR
    # masked_la = torch.stack(list(filter(lambda x: x is not None, [i[5] for i in data]))) # remove for XTR
    # lm_position = torch.stack(list(filter(lambda x: x is not None, [i[6] for i in data])))
    guid = torch.stack(list(filter(lambda x: x is not None, [i[7] for i in data])))
    tgt_la1 = torch.stack(list(filter(lambda x: x is not None, [i[8] for i in data])))
    tgt_la2 = torch.stack(list(filter(lambda x: x is not None, [i[9] for i in data])))
    # return sent1, sent1_len, sent2, sent2_len, lm_label, masked_la, tgt_la1, tgt_la2
    return sent1, sent1_len, sent2, sent2_len, torch.tensor([]), torch.tensor([]), tgt_la1, tgt_la2


class DataProvider(object):
    def __init__(self, config, shuffle=True, model_type='train', make_train=True, make_valid=True, provide_valid=False):
        # randomly select samples to compose validation set.
        logger.info("Reading from : " + config.train_data_path)
        if make_train:
            self.dataset = InputDataset(config.train_data_path, config.validation_size, config, config.bpe_path, config.max_seq_len, make_valid, provide_valid)
            self.tokenizer = self.dataset.tokenizer 
        if make_valid:
            if provide_valid:
                self.vali_dataset = ValiDataset(None, config.valid_data_path, config, config.bpe_path, config.max_seq_len, provide_valid)
            else:
                self.vali_dataset = ValiDataset(self.dataset.vali_inputs, None, config, config.bpe_path, config.max_seq_len, provide_valid)
            if not make_train:
                self.tokenizer = self.vali_dataset.tokenizer
        # self.data_loader = DataLoader(self.dataset, config.train_batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=True)
        # self.vali_loader = DataLoader(self.vali_dataset, config.vali_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

