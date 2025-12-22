from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
# from config import Config
from configs.best_config_SHD_former_TIM_Spikformer import Config as SHDTConfig
from module.spikformer_TIM import Spikformer

import utils
import numpy as np
import torch
from utils import init_logger,build_optimizer
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

def calc_loss(config, output, y):

    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(output, y)

        return loss

def calc_loss_nonspike(config, output, y):
    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(output, y)

        return loss

def calc_metric(config, output, y):

    return np.mean((torch.max(y,1)[1]==torch.max(output,1)[1]).detach().cpu().numpy())




def eval_model(config, model, loader, device):
    ##################################    Eval Loop    #########################
    model.eval()
    # calc_loss_std = SoftTargetCrossEntropy()
    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y, _) in enumerate(tqdm(loader)):
            y = F.one_hot(y, config.n_outputs).float()
            if config.use_padding:
                current_time = x.size(1)
                target_time = config.max_len

                
                padding_needed = max(0, target_time - current_time) 

                padding = (0, 0, 0, padding_needed)

                # 应用填充
                x = F.pad(x, padding, 'constant', 0)
            x = x.float().to(device)
            y = y.to(device)

            output = model(x)

            loss = calc_loss(config, output, y)
            
            # loss = calc_loss_std(output,y)
            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

    loss_valid = np.mean(loss_batch)
    metric_valid = np.mean(metric_batch)
    return loss_valid, metric_valid



def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):

    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    best_loss_val = 1e6
    # calc_loss_std = SoftTargetCrossEntropy()

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []
        for i, (x, y, x_len) in enumerate(tqdm(train_loader)):

            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output= model(x)
            
            loss = calc_loss(config, output, y)
            

            loss.backward()
            optimizer.step()

            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        scheduler.step()

        # best_model_wts = copy.deepcopy(model.state_dict())

        ##################################    Eval Loop    #########################
        model.eval()
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (x, y, x_len) in enumerate(tqdm(valid_loader)):
                y = F.one_hot(y, config.n_outputs).float()

                x = x.float().to(device)
                y = y.to(device)

                output = model(x)
                
                loss = calc_loss(config, output, y)
                
                metric = calc_metric(config, output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                functional.reset_net(model)

        loss_valid = np.mean(loss_batch)
        metric_valid = np.mean(metric_batch)

        loss_epochs['valid'].append(loss_valid)
        metric_epochs['valid'].append(metric_valid)
        #
        if test_loader:
            loss_test, metric_test = eval_model(config, model, test_loader, device)
        else:
            # could be improved
            loss_test, metric_test = 100, 0
        #
        loss_epochs['test'].append(loss_test)
        metric_epochs['test'].append(metric_test)

        ########################## Logging and Plotting  ##########################


        logger.info(
            f"=====> Epoch {epoch} : Loss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100 * metric_epochs['train'][-1]:.2f}%")
        logger.info(
            f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100 * metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100 * max(metric_epochs['valid'][-1], best_metric_val):.2f}%")
        if test_loader:
            logger.info(
                f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100 * metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100 * max(metric_epochs['test'][-1], best_metric_test):.2f}%")

        checkpoint_dir = os.path.join('./checkpoints', config.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if metric_valid > best_metric_val:  # and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = metric_valid

        if loss_valid < best_loss_val:  # and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
            print("# Saving best Loss model...")
            torch.save(model.state_dict(),ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

        if metric_test > best_metric_test:  # and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
            best_metric_test = metric_test

from models.spikescr import SpikeDrivenTransformer






if __name__ == '__main__':


    config = SHDTConfig()
    # config = GSCTConfig()
    # config = SSCTConfig()

    logger = init_logger(config, "training")
    logger.info("Logger is properly initialized and ready to use.")
    logger.info("The GPU is {}".format(config.gpu))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = config.gpu
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    print()
    print(f"\n=====> Device = {device} \n\n")

    """ dataset """
    if config.dataset == 'shd':
        train_loader, valid_loader = SHD_dataloaders(config)
        test_loader = None
    elif config.dataset == 'ssc':
        train_loader, valid_loader, test_loader = SSC_dataloaders(config)
    elif config.dataset == 'gsc':
        train_loader, valid_loader, test_loader = GSC_dataloaders(config)
    else:
        raise Exception(f'dataset {config.dataset} not implemented')



    
    # for hidden_dim in config.n_hidden_neurons_list:

    ''' set random seeds '''
    seed_val = config.seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("##############################################\n")
    logger.info("Seed :{}".format(seed_val))
    epochs = config.epochs
    logger.info("The epoch is: {}".format(epochs))
    logger.info("The batch size is : {}".format(config.batch_size))

    if config.use_aug:
        logger.info("The TN_mask_aug_proba is: {}".format(config.TN_mask_aug_proba))
        # logger.info("The time_mask_size is: {}".format(config.time_mask_size))
        logger.info("The time_mask_proportion is :{}".format(config.time_mask_proportion))
        logger.info("The neuron_mask_size is: {}".format(config.neuron_mask_size))

    else:
        logger.info("The augs is None")

    """ dataset """
    if config.dataset == 'shd':
        train_loader, valid_loader = SHD_dataloaders(config)
        test_loader = None
    elif config.dataset == 'ssc':
        train_loader, valid_loader, test_loader = SSC_dataloaders(config)
    elif config.dataset == 'gsc':
        train_loader, valid_loader, test_loader = GSC_dataloaders(config)

    else:
        raise Exception(f'dataset {config.dataset} not implemented')


    model = Spikformer(config, num_classes=config.n_outputs, depths=config.depths).to(device)


    logger.info("Model size:{}".format(utils.count_parameters(model)))
    lr_w = config.lr_w
    logger.info("lr_w: {}".format(lr_w))
    logger.info("weight_decay:{}".format(config.weight_decay))
    optimizer = build_optimizer(config, model)
    T = config.t_max
    logger.info("T:{}".format(T))

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")  
    dataset_info = config.dataset
    # 指定文件夹路径
    folder_path = os.path.join('model_structure', dataset_info)
    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)

    # 构造文件名并包括路径
    filename = os.path.join(folder_path, f'model_structure_{dataset_info}_{formatted_time}.txt')

    with open(filename, 'w') as f:
        # 将print函数的输出临时重定向到文件
        print(model, file=f)

    print(f"===> Dataset    = {config.dataset}")
    logger.info(f"===> Model type = {config.model_type}")
    print(f"===> Model size = {utils.count_parameters(model)}\n\n")

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, epochs)