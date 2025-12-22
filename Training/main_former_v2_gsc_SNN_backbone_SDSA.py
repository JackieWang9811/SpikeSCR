from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from configs.best_config_GSC_former_SDSA import Config as GSCTConfig
from module.spike_driven_tranformer import SpikeDrivenTransformer
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
from datasets import SpecAugment

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

def calc_loss(config, output, y):

    # probably better to add it in init, or in general do it one time only
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

    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y, x_len) in enumerate(tqdm(loader)):
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
            # x for shd and ssc is: (batch, time, neurons)
            # x={Tensor:(256, 101,,140)}

            if config.use_aug:
                x = augs(x,x_len)

            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons) => (512,101,140)
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



if __name__ == '__main__':


    config = GSCTConfig()
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

    augs = SpecAugment(config)

    for hidden_dim in config.n_hidden_neurons_list:

        ''' set random seeds '''
        seed_val = config.seed
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cp.random.seed(seed_val)
        logger.info("##############################################\n")
        logger.info("Seed :{}".format(seed_val))
        epochs = config.epochs
        logger.info("The epoch is: {}".format(epochs))
        logger.info("The batch size is : {}".format(config.batch_size))
        logger.info("The backend is: {}".format(config.backend))
        logger.info("The num_heads is: {}".format(config.num_heads))

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

        config.n_hidden_neurons = hidden_dim
        config.hidden_dims = 4*hidden_dim

        if config.use_aug:
            logger.info("The mF is: {}".format(config.mF))
            logger.info("The F is: {}".format(config.F))
            logger.info("The mT is: {}".format(config.mT))
            logger.info("The pS is: {}".format(config.pS))

        else:
            logger.info("The augs is None")

        logger.info("The window_size is {}".format(config.window_size))
        logger.info("The hop_length is {}".format(config.hop_length))
        logger.info("The n_mels is {}".format(config.n_mels))
        logger.info("The hidden_dim is: {}".format(hidden_dim))
        logger.info("The spike_mode is: {}".format(config.spike_mode))
        logger.info("The block_mode is :{}".format(config.block_type))
        logger.info("The gate_v_threshold is: {}".format(config.gate_v_threshold))
        model = SpikeDrivenTransformer(config, num_classes=config.n_outputs).to(device)

        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M%S")
        dataset_info = config.dataset
        folder_path = os.path.join('model_structure', dataset_info)
        os.makedirs(folder_path, exist_ok=True)

        filename = os.path.join(folder_path, f'model_structure_{dataset_info}_{formatted_time}.txt')

        with open(filename, 'w') as f:
            print(model, file=f)

        print(f"===> Dataset    = {config.dataset}")
        print(f"===> Model type = {config.model_type}")
        print(f"===> Model size = {utils.count_parameters(model)}\n\n")
        logger.info("Model size:{}".format(utils.count_parameters(model)))
        lr_w = config.lr_w
        logger.info("lr_w: {}".format(lr_w))
        weight_decay = config.weight_decay
        logger.info("weight_decay: {}".format(weight_decay))
        optimizer = build_optimizer(config, model)
        T = config.t_max
        logger.info("T:{}".format(T))
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

        train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, cosine_scheduler, epochs)