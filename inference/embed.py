import os
import io
import sys
from contextlib import ExitStack
from logging import getLogger
import numpy as np
import torch
import shutil
import argparse
import copy
import logging
import pickle
import random
import jieba
from sacremoses import MosesTokenizer
from collections import OrderedDict
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
from tqdm import tqdm
from config import config
sys.path.append('..')
import torch.nn.functional as F
from modeling import BilingualModel
from preprocessing import BPETokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils import *
import glob
import opencc
from pyknp import Juman
converter = opencc.OpenCC('t2s.json')                       
special_la = ['zh', 'ja', 'wuu', 'yue']
jumanpp = Juman()


def load_data(data_path, la='en', lower=True):
    if la not in special_la:
        mosetokenizer = MosesTokenizer(la)
    data = []
    with ExitStack() as stack:
        f = stack.enter_context(open(data_path))
        for i in tqdm(f, desc='Loading data..'):
            i = i.lower().strip() if lower else i.strip()
            if i != '':
                if la in special_la and la != 'ja':
                    data.append(' '.join(jieba.cut(i)))
                elif la == 'ja':
                    data.append(' '.join([mrph.midasi for mrph in jumanpp.analysis(i).mrph_list()]))
                else:
                    data.append(' '.join(mosetokenizer.tokenize(i, escape=False)))
    logger.info('tokenzied sample: {}'.format(data[random.randint(0, len(data) - 1)]))
    data = np.array(data)
    return data


def pad_seq(seq, length):
    return F.pad(seq, pad=(0, length - seq.shape[-1]), mode='constant', value=0)


def collate_fn_eval(data):
    max_len= max([i[1] for i in data])
    sent = torch.stack([pad_seq(i[0],max_len) for i in data])
    sent_len = torch.stack([i[1] for i in data])
    return sent, sent_len


class EvalDataset(Dataset):
    def __init__(self, data, bpe_path='bpe.model', max_len=120):
        self.tokenizer = BPETokenizer(bpe_path, max_len)
        self.vocab_size = len(self.tokenizer.sp)
        self.max_len = max_len
        self.inputs = data
        self.num_samples = len(data)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, item):

        raw_input = self.inputs[item]
        ids = self.tokenizer.convert_sents_to_ids(raw_input)
        cur_tensors = (torch.tensor(ids),
                       torch.tensor(len(ids)))
        return cur_tensors


class DataProvider(object):
    def __init__(self, config, la, eval_data, is_path=True, shuffle=True, model_type='eval'):
        if is_path:
            data = load_data(eval_data, la, config.lower)
        else:
            data = np.array(eval_data)
        self.dataset = EvalDataset(data, config.bpe_path, config.max_seq_len)
        self.tokenizer = self.dataset.tokenizer
        self.dataloader = DataLoader(self.dataset, config.eval_batch_size, shuffle= False, \
                                                 collate_fn=collate_fn_eval)

        
class Evaluator(object):
    """
    Evaluator:
    1. Load tokenizer to tokenize the raw sentences
    2. Load model parameters
    3. Encode tokenized sentences into sentence embeddings
    """
    def __init__(self, config, device='cpu'):
        self.device = device
        self.config = config
    
    def fix_model_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
        return new_state_dict
        
    def load_model(self, model_path):
        """
        A function to resume parameters from model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        old_state = checkpoint['state_dict']
        old_state = self.fix_model_state_dict(old_state)
         
        model = BilingualModel(self.config.vocab_size, self.config)
        try:
            model.load_state_dict(old_state)
        except:
            old_names = ['model.decoder.transform.dense.weight', 'model.decoder.transform.dense.bias', 'model.decoder.transform.LayerNorm.weight', 'model.decoder.transform.LayerNorm.bias']
            new_names = ['model.transform.dense.weight', 'model.transform.dense.bias', 'model.transform.LayerNorm.weight', 'model.transform.LayerNorm.bias']
            for k1,k2 in zip(old_names, new_names):
                old_state[k2] = old_state[k1]
                del old_state[k1]
            model.load_state_dict(old_state)
        model.eval() 
        loss = checkpoint['best_loss']
        step = checkpoint['step']
        print('step {} with best loss: {}'.format(step, loss))
        return model
        
        
    def get_sentence_embeddings(self, provider, model):
        """
        A function to run the model to get corresponding embeddings of given text.
        """
        with torch.no_grad():
            for i, batch in tqdm(enumerate(provider.dataloader)):
                batch = tuple(t.to(device=self.device) for t in batch)
                sent_ids, sent_len = batch
                output = model.inference(sent_ids, sent_len)
                if i == 0:
                    embeddings = output
                else:
                    embeddings = torch.cat((embeddings, output), 0)
        return embeddings
 

def encode(data, model_path='../ckpt/EMS_model.pt', model_name='EMS', la='en', device='cpu'):
    logger.info('language: {}'.format(la))
    if la == 'none':
        data = [line.lower().strip() for line in data]
    elif la not in special_la:
        mosetokenizer = MosesTokenizer(la)
        data = [' '.join(mosetokenizer.tokenize(line.lower().strip(), escape=False)) for line in data]
    elif la == 'ja':
        data = [' '.join([mrph.midasi for mrph in jumanpp.analysis(line.lower().strip()).mrph_list()]) for line in data]
    else:
        data = [' '.join(jieba.cut(line.lower().strip())) for line in data]
    logger.info('tokenzied sample: {}'.format(data[random.randint(0, len(data) - 1)]))
    config.set_eval()
    provider = DataProvider(config, la, data, False, False, 'eval')
    config.set_config(model_name, resume=False, is_train=False)
    evaluator = Evaluator(config, device)
    logger.info('current model name: ' + config.model_name)
    model = evaluator.load_model(model_path)
    model.to(evaluator.device)
    logger.info('Inferencing...')
    embeddings = evaluator.get_sentence_embeddings(provider, model)
    return embeddings


def main(data_path, model_path, model_name, la, save_path, device='cpu'):
    logger.info('language: {}'.format(la))
    config.set_eval()
    provider = DataProvider(config, la, data_path, True, False, 'eval')
    config.set_config(model_name, resume=False, is_train=False)
    evaluator = Evaluator(config, device)
    logger.info('current model name: ' + config.model_name)
    model = evaluator.load_model(model_path)
    model.to(evaluator.device)
    logger.info('Inferencing...')
    embeddings = evaluator.get_sentence_embeddings(provider, model)
    emb = embeddings.to('cpu').detach().numpy().copy()
    with open(save_path, 'wb') as f:
        pickle.dump(emb, f)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='language', default='en', help='language', required=True, type=str)
    parser.add_argument('-m', dest='model_name', default='EMS', help='model name', type=str)
    parser.add_argument('--model_path', dest='model_path', default='../ckpt/EMS_model.pt', help='path of the model', type=str)
    parser.add_argument('--data_path', dest='data_path', default=None, help='path of the data', required=True, type=str)
    parser.add_argument('--save_to', dest='save_path', default=None, help='path of the binarized embeddings to save', type=str)
    parser.add_argument('--gpu', dest='gpu', help='use gpu or not', action='store_true')
    args = parser.parse_args()
    if args.gpu:
        torch.cuda.set_device(0)
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'
    main(args.data_path, args.model_path, args.model_name, args.language, args.save_path, device)

