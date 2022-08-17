#coding:utf8
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from shutil import copyfile
import random
from io import open
import os
from torch.autograd import Variable
import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import detect_anomaly
from tqdm import tqdm, trange
from modeling import BilingualModel
from preprocessing import DataProvider
from preprocessing import dataset
from modeling.optimization import BertAdam, warmup_linear
from torch.utils.data import Dataset
from utils import *
from config import config 
import logging
import time
from datetime import datetime
from torch.cuda import amp
import pytorch_warmup as warmup
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)), size=self.num_samples,
                p=self.weights.numpy() / torch.sum(self.weights).numpy(), replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def reduce_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def evaluate(val_loader, model, epoch, device):
    val_progressor = ProgressBar(mode="Vali",\
                                 epoch=epoch,\
                                 total_epoch=config.num_train_epochs,\
                                 model_name=config.model_name,\
                                 total=len(val_loader))
    model.to(device)
    logging_valid_loss = AverageMeter()
    logging_valid_loss1 = AverageMeter()
    logging_valid_loss2 = AverageMeter()
    logging_valid_acc = AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_progressor.current = i 
            batch = tuple(t.to(device) for t in batch)
            sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la, tgt_la1, tgt_la2 = batch
            loss, xtr_loss, sentalign_loss, acc = model(
                    sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la, tgt_la1, tgt_la2)
            logging_valid_loss.update(torch.mean(loss).item(), sent1_ids.shape[0])
            logging_valid_loss1.update(torch.mean(xtr_loss).item(), sent1_ids.shape[0])
            logging_valid_loss2.update(torch.mean(sentalign_loss).item(), sent1_ids.shape[0])
            logging_valid_acc.update(torch.mean(acc).item(), sent1_ids.shape[0])
            val_progressor.current_loss = logging_valid_loss.val
            val_progressor.avg_loss = logging_valid_loss.avg
            val_progressor.current_loss1 = logging_valid_loss1.val
            val_progressor.avg_loss1 = logging_valid_loss1.avg
            val_progressor.current_loss2 = logging_valid_loss2.val
            val_progressor.avg_loss2 = logging_valid_loss2.avg
            val_progressor.cur_time = str(datetime.now().strftime('%d %H:%M:%S'))
            if i % 500 == 0:
                val_progressor()
        val_progressor.done()
    return logging_valid_loss.avg

def main():
    # os.environ['NCCL_SOCKET_IFNAME'] = args.ifname
    # device = torch.device('cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else "cpu")
    # args.ngpu_per_node = torch.cuda.device_count()
    # args.rank = int(os.environ["RANK"])
    # if args.world_size != args.ngpu_per_node:
    args.rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.rank)
    torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend="nccl", init_method='tcp://'+args.mainip+':'+args.port, world_size=args.world_size, rank=args.rank)
    # else:
        # args.rank = args.local_rank
        # torch.distributed.init_process_group(backend="nccl")

    if args.rank == 0:
        if not config.resume:
            if not os.path.exists(config.output_dir):
                os.mkdir(config.output_dir)
            if os.path.exists(os.path.join(config.output_dir, config.model_name)):
                if os.path.exists(os.path.join(config.output_dir, config.model_name, config.best_models)) and os.listdir(os.path.join(config.output_dir, config.model_name, config.best_models)):
                    logger.error('Exists best checkpoints in {}. Please Check.'.format(os.path.join(config.output_dir, config.model_name,  config.best_models)))
                    exit(0)
            if os.path.exists(os.path.join(config.output_dir, config.model_name)):
                shutil.rmtree(os.path.join(config.output_dir, config.model_name))
            os.mkdir(os.path.join(config.output_dir, config.model_name))
            os.mkdir(os.path.join(config.output_dir, config.model_name, config.best_models))
            shutil.copy(config.config_path, os.path.join(config.output_dir, config.model_name, 'config_'+config.model_name +'.py'))
            logger.info('Copying config.py to {}'.format(os.path.join(config.output_dir, config.model_name, 'config_'+config.model_name)))
        
    # device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda", args.rank)
    model = BilingualModel(config.vocab_size, config)
    # if torch.cuda.device_count() > 1:
    model.to(device)
    model = DDP(model, device_ids=[args.rank], output_device=args.rank)

        # model = model.cuda()
        # device_ids = list(range(torch.cuda.device_count()))
        # model = nn.DataParallel(model, device_ids=device_ids)
    #else:
        # model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, amsgrad=True)

    start_epoch = 1
    best_loss = np.inf
    # best_loss_save = np.inf
    resume = config.resume
    
    if args.rank == 0:
        if resume:
            if not os.path.exists(os.path.join(config.output_dir, config.model_name)):
                logger.error('No model found in path: {}'.format(os.path.join(config.output_dir, config.model_name)))
            checkpoint = torch.load(os.path.join(config.output_dir, config.model_name, config.best_models, 'model_best.pt'))
            old_state = checkpoint['state_dict']
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(old_state)
            optimizer.load_state_dict(checkpoint["optimizer"])
    
    # dataloader = DataProvider(config, True, 'train')
    # train_dataset = dataloader.dataset
    if args.rank == 0:
        logger.info("Loading valid data ... %s", args.valid_path)
        validation_dataset = torch.load(args.valid_path)
        validation_data = DataLoader(validation_dataset, batch_size=config.vali_batch_size, shuffle=False, collate_fn=dataset.collate_fn, drop_last=False)
    
    # config.train_data_path = "data/62languages.uncased.train.abci." + str(args.rank) + ".shuf"
    # train_dataset = DataProvider(config, True, "train", True, False, False).dataset

    logger.info("Loading train data for GPU:%d ... %s", args.rank, args.train_path + "." + str(args.rank) + ".pt")
    train_dataset = torch.load(args.train_path + "." + str(args.rank) + ".pt") # !formal!
    # train_dataset = torch.load(args.train_path + ".0.pt") #
    # train_dataset = torch.load(args.train_path + ".tmp5." + str(args.rank) + ".pt") # debug
    # train_dataset = torch.load(args.train_path + ".tmp.pt")
    # train_dataset = torch.load(args.train_path + ".0.pt")
    # train_dataset = torch.load(args.train_path)
    
    logger.info("Preparing train data for GPU:%d ...", args.rank)
    if config.do_oversampling:
        # print(train_dataset.weights[0], train_dataset.weights[1])
        sampler = CustomWeightedRandomSampler(weights=train_dataset.weights, num_samples=config.num_sample, replacement=True)
        train_data = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=dataset.collate_fn, drop_last=True, sampler=sampler)
    elif config.ddp_sampler:
        sampler = DistributedSampler(train_dataset)
        train_data = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=dataset.collate_fn, sampler=sampler)
    else:
        train_data = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)   
    
    # train_data = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, \
    #                        collate_fn=dataset.collate_fn, sampler=sampler, shuffle=False, \
    #                        num_workers=32, pin_memory=True, drop_last=True)
    # dist.barrier()

    num_train_optimization_steps = len(train_data) * config.num_train_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_optimization_steps)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.5)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10000)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # warmup_scheduler.last_step = -1
    global_step = 0
    num_param = sum([param.nelement() for param in model.parameters()])
    
    if args.rank == 0:
        logger.info("***** Running Training *****")
        logger.info("  Model Name = %s", config.model_name)
        logger.info("  Number of the parameters: %d", num_param)
        logger.info("  bpe_model_path = %s", config.bpe_path)
        logger.info("  Num examples = %d on each GPU", train_data.dataset.num_samples)
        logger.info("  Batch size = %d on each GPU", config.train_batch_size)
        logger.info("  Num steps per epoch = %d on each GPU", int(len(train_data)))
        logger.info("  Has language embedding: %s", str(config.has_la_emb))
        logger.info("  Share embedding layer weights: %s", str(config.share_emb))
        logger.info("  Generative task: %s", config.generative_task)
        logger.info("  Has sentence-alignment loss: %s", str(config.has_sentence_loss))
        logger.info("  Has sentence-similarity loss: %s", str(config.has_sentence_similarity_loss))
    
    model.train()
    scaler = amp.GradScaler()
    
    for epoch in range(start_epoch, int(config.num_train_epochs) + 1):
        dist.barrier()
        if config.ddp_sampler:
            sampler.set_epoch(epoch)
        
        logger.info('Start new epoch %s on GPU %d', str(epoch), args.rank)
        if args.rank == 0:
            logging_loss = AverageMeter()
            logging_loss1 = AverageMeter()
            logging_loss2 = AverageMeter()
            train_progressor = ProgressBar(mode="Train", epoch=epoch,
                                            total_epoch=int(config.num_train_epochs),
                                            model_name=config.model_name,
                                            total=len(train_data),
                                            )
        
        if args.rank == 0 and epoch == 1:
            if config.has_validation:
                valid_loss = evaluate(validation_data, model, epoch, device)
                eval_loss = valid_loss
                best_loss = valid_loss
                is_best = False
                logger.info("Validation successful! Initial validation loss: %f", valid_loss)

        steps_per_epoch = len(train_data)
        
        for step, batch in enumerate(train_data):
            dist.barrier()
            
            batch = tuple(t.to(device) for t in batch)
            sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la, tgt_la1, tgt_la2 = batch
            
            with amp.autocast():
                loss, xtr_loss, sentalign_loss, recall = model(
                        sent1_ids, sent1_len, sent2_ids, sent2_len, lm_label_id, masked_la, tgt_la1, tgt_la2
                        )
            
            dist.barrier()
            reduced_loss = reduce_mean(loss.data)
            reduced_xtr_loss = reduce_mean(xtr_loss.data)
            reduced_sentalign_loss = reduce_mean(sentalign_loss.data)
            if args.rank == 0:
                logging_loss.update(reduced_loss.item(), sent1_ids.shape[0])
                logging_loss1.update(reduced_xtr_loss.item(), sent1_ids.shape[0])
                logging_loss2.update(reduced_sentalign_loss.item(), sent1_ids.shape[0])
                train_progressor.current = step + 1
                train_progressor.current_loss = logging_loss.val
                train_progressor.current_loss1 = logging_loss1.val
                train_progressor.current_loss2 = logging_loss2.val
                train_progressor.last_lr = optimizer.param_groups[0]['lr']
                train_progressor.avg_loss = logging_loss.avg
                train_progressor.avg_loss1 = logging_loss1.avg
                train_progressor.avg_loss2 = logging_loss2.avg
                train_progressor.cur_time = str(datetime.now().strftime('%d %H:%M:%S'))
            if step % 1000 == 0 and args.rank == 0:
                train_progressor()
                # logger.info("GPU: %d", args.rank)
            
            if config.step_checkpoint is True:
                if (step + 1) % (int(steps_per_epoch /
                    config.num_checkpoint_per_epoch)) == 0 and args.rank == 0:
                    if config.has_validation:
                        valid_loss = evaluate(validation_data, model, epoch, device)
                        eval_loss =  valid_loss
                        is_best = eval_loss <= best_loss
                        best_loss = min(eval_loss, best_loss)
                    
                    logger.info('Trying to save model.')
                    save_checkpoint(
                            {
                                "epoch": epoch,
                                "step": step + 1,
                                "model_name": config.model_name,
                                "state_dict": model.state_dict(),
                                "best_loss": best_loss,
                                "optimizer": optimizer.state_dict(),
                                "valid_loss": eval_loss,
                                }, is_best, epoch, step + 1,
                            )
            
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            global_step += 1 # ???
            scaler.update()
            lr_scheduler.step(lr_scheduler.last_epoch + 1) # UserWarning
            warmup_scheduler.dampen()
        
        if args.rank == 0:
            train_progressor.done()
        
        # torch.cuda.synchronize()

        # eval_loss = loss
        if config.epoch_checkpoint is True and config.step_checkpoint is False:
            if args.rank == 0:
                if config.has_validation:
                    valid_loss = evaluate(validation_data, model, epoch, device)
                    eval_loss = valid_loss
                    is_best = eval_loss <= best_loss
                    best_loss = min(eval_loss, best_loss)
            
                logger.info('Trying to save model.')
                save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_name": config.model_name,
                            "state_dict": model.state_dict(),
                            "best_loss": best_loss,
                            "optimizer": optimizer.state_dict(),
                            "valid_loss": eval_loss,
                            }, is_best, epoch,
                        )

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--ngpu_per_node", type=int, default=-1)
    parser.add_argument('--model', dest='model_name', default='', help='modelName')
    parser.add_argument('--resume', dest='resume', default=False, help='resume training')
    parser.add_argument('--is_train', dest='is_train', default=True, help='is train?')
    # parser.add_argument('--la', dest='target_la', default='fr', help='target language')
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--valid_path", type=str, default="")
    parser.add_argument("--port", type=str, default="")
    parser.add_argument("--mainip", type=str, default="")
    parser.add_argument("--ifname", type=str, default="")
    args = parser.parse_args()    
    config.set_config(args.model_name, args.resume, args.is_train)
    main()

