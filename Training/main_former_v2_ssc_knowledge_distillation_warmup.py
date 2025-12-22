from datasets import SHD_dataloaders, SSC_dataloaders_teacher_student, GSC_dataloaders
from configs.best_config_SSC_former import Config as SSCTConfig
from configs.best_config_SSC_former_teacher import Config as TSSCTConfig

from models.spikescr import SpikeDrivenTransformer
from spikingjelly.datasets import padded_sequence_mask
from module.scheduler import WarmUpLR, get_scheduler

import utils
import numpy as np
import torch
from utils import init_logger,build_optimizer
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
    if config.loss == 'mean':
        m = torch.mean(output, 0)
    elif config.loss == 'max':
        m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count':
        m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(m, y)
        # log_softmax_fn = nn.LogSoftmax(dim=1)
        # loss_fn = nn.NLLLoss()
        # log_p_y = log_softmax_fn(m)
        # loss = loss_fn(log_p_y, y)

        return loss

def calc_loss_nonspike(config, output, y):
    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(output, y)
        # log_softmax_fn = nn.LogSoftmax(dim=1)
        # loss_fn = nn.NLLLoss()
        # log_p_y = log_softmax_fn(m)
        # loss = loss_fn(log_p_y, y)

        return loss

def calc_metric(config, output, y):
    # mean accuracy over batch
    if config.loss == 'mean': m = torch.mean(output, 0)
    elif config.loss == 'max': m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count': m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

        return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())


def calc_distillation_loss(output, target_output, temperature=2.0):
    """
    Calculate the distillation loss as the KL divergence between the
    softened outputs of the current model and the target outputs from the
    teacher model, using temperature scaling.
    """
    # soft_output = F.log_softmax(output / temperature, dim=1)
    # soft_target = F.softmax(target_output / temperature, dim=1)
    # return F.kl_div(soft_output, soft_target, reduction='batchmean') * (temperature ** 2)

    # Apply softmax and sum as per the model's output handling
    softmax_fn = nn.Softmax(dim=2)
    processed_output = torch.sum(softmax_fn(output), 0)
    processed_target_output = torch.sum(softmax_fn(target_output), 0)

    # Soften the outputs with the temperature
    soft_output = F.log_softmax(processed_output / temperature, dim=1)
    soft_target = F.softmax(processed_target_output / temperature, dim=1)

    # Calculate the KL divergence, scale by temperature squared
    return F.kl_div(soft_output, soft_target, reduction='batchmean') * (temperature ** 2)

def dynamic_calc_distillation_loss(student_logits, teacher_logits, temperature=1.0):

    softmax_fn = nn.Softmax(dim=2)
    student_logits = torch.sum(softmax_fn(student_logits), 0)
    teacher_logits = torch.sum(softmax_fn(teacher_logits), 0)

    # with torch.no_grad():
    student_probs = F.softmax(student_logits, dim=-1)
    student_entropy = - torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1) # student entropy, (bsz, )
    # normalized entropy score by student uncertainty:
    # i.e.,  entropy / entropy_upper_bound
    # higher uncertainty indicates the student is more confusing about this instance
    instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))

    input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    batch_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
    weighted_kld = torch.mean(batch_loss * instance_weight)

    return weighted_kld

def normal_calc_distillation_loss(student_logits, teacher_logits, temperature=1.0):

    softmax_fn = nn.Softmax(dim=2)
    student_logits = torch.sum(softmax_fn(student_logits), 0)
    teacher_logits = torch.sum(softmax_fn(teacher_logits), 0)

    # # with torch.no_grad():
    # student_probs = F.softmax(student_logits, dim=-1)
    # student_entropy = - torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1) # student entropy, (bsz, )
    # # normalized entropy score by student uncertainty:
    # # i.e.,  entropy / entropy_upper_bound
    # # higher uncertainty indicates the student is more confusing about this instance
    # instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))

    input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    batch_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
    weighted_kld = torch.mean(batch_loss)

    return weighted_kld


def eval_model(config, model, loader, device):
    ##################################    Eval Loop    #########################
    model.eval()
    # calc_loss_std = SoftTargetCrossEntropy()
    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y, x_len) in enumerate(tqdm(loader)):
            # x for shd and ssc is: (batch, time, neurons)
            # x={Tensor:(256, 101,,140)}
            attention_mask = padded_sequence_mask(x_len)
            attention_mask = attention_mask.transpose(0,1).to(device)
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

            output = model(x,attention_mask)

            loss = calc_loss(config, output, y)
            
            # loss = calc_loss_std(output,y)
            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

        # if self.config.DCLSversion == 'gauss' and self.config.model_type != 'snn':
        #     for i in range(len(self.blocks)):
        #         self.blocks[i][0][0].version = 'gauss'
        #         self.blocks[i][0][0].DCK.version = 'gauss'

        # model.load_state_dict(torch.load(eventid + '.pt'), strict=True)
        # if os.path.exists(eventid + '.pt'):
        #     os.remove(eventid + '.pt')
        # else:
        #     print(f"File '{eventid + '.pt'}' does not exist.")
    loss_valid = np.mean(loss_batch)
    metric_valid = np.mean(metric_batch)
    return loss_valid, metric_valid



def train_model(config, train_loader, teacher_train_loader, valid_loader, test_loader, device, model, teacher_model, optimizer, schedulers, num_epochs):

    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    best_loss_val = 1e6
    # calc_loss_std = SoftTargetCrossEntropy()

    teacher_iter = iter(teacher_train_loader)

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        teacher_model.eval()  # 确保教师模型处于评估模式

        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []
        # max_t = 0
        # pre_pos = pre_pos_epoch.copy()
        for i, (x, y, x_len) in enumerate(tqdm(train_loader)):
            # x for shd and ssc is: (batch, time, neurons)
            # x={Tensor:(256, 101,,140)}

            try:
                teacher_x, teacher_y, teacher_x_len = next(teacher_iter)
            except StopIteration:
                teacher_iter = iter(teacher_train_loader)
                teacher_x, teacher_y, teacher_x_len = next(teacher_iter)

            attention_mask = padded_sequence_mask(x_len)
            attention_mask = attention_mask.transpose(0,1).to(device)

            # if self.config.augment:
            #     x, y = augmentations(x, y)
            # current_max_t = x.shape[1]  # 获取当前批次中的最大时间步长
            # if current_max_t > max_t:
            #     max_t = current_max_t  # 更新最大时间步长
            if config.use_padding:
                current_time = x.size(1)
                target_time = config.max_len

                
                padding_needed = max(0, target_time - current_time) 

                padding = (0, 0, 0, padding_needed)

                # 应用填充
                x = F.pad(x, padding, 'constant', 0)

            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons) => (512,101,140)
            y = y.to(device)

            # 使用教师模型生成目标输出
            with torch.no_grad():
                teacher_attention_mask = padded_sequence_mask(teacher_x_len).transpose(0, 1).to(device)
                teacher_x = teacher_x.float().to(device)
                teacher_output = teacher_model(teacher_x, teacher_attention_mask)

            # zero the parameter gradients
            optimizer.zero_grad()

            output= model(x,attention_mask)
            
            ce_loss = calc_loss(config, output, y)
            

            # 计算蒸馏损失
            distillation_loss = normal_calc_distillation_loss(output, teacher_output, temperature=1.0)

            # 组合损失
            loss = 1.0 * ce_loss + 0.5 * distillation_loss  # 调整蒸馏损失的权重

            loss.backward()
            optimizer.step()

            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)
            functional.reset_net(teacher_model)

        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        if schedulers["warmup"] is not None and epoch < config.n_warmup:
            schedulers["warmup"].step()

        elif schedulers["scheduler"] is not None:
            schedulers["scheduler"].step()

        # best_model_wts = copy.deepcopy(model.state_dict())

        ##################################    Eval Loop    #########################
        model.eval()
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (x, y, x_len) in enumerate(tqdm(valid_loader)):
                attention_mask = padded_sequence_mask(x_len)
                attention_mask = attention_mask.transpose(0, 1).to(device)

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

                output = model(x,attention_mask)
                
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

        checkpoint_dir = os.path.join('./checkpoints/progressive_kd', config.dataset)
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


    # config = SHDTConfig()
    # config = GSCTConfig()
    config = SSCTConfig()
    teacher_config = TSSCTConfig()
    config.log_dir = './logs/logging/ssc/PTCD'
    
    logger = init_logger(config,"training")
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

    seed_val = config.seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

        
    for hidden_dim in config.n_hidden_neurons_list:

        ''' set random seeds '''

        # cp.random.seed(seed_val)
        logger.info("##############################################\n")
        logger.info("Seed :{}".format(seed_val))
        epochs = teacher_config.epochs
        logger.info("The epoch is: {}".format(epochs))
        logger.info("The batch size is : {}".format(config.batch_size))
        logger.info("The backend is: {}".format(config.backend))
        logger.info("The num_heads is: {}".format(config.num_heads))

        if config.use_aug:
            logger.info("The TN_mask_aug_proba is: {}".format(config.TN_mask_aug_proba))
            logger.info("The time_mask_proportion is: {}".format(config.time_mask_proportion))
            logger.info("The neuron_mask_size is: {}".format(config.neuron_mask_size))

        else:
            logger.info("The augs is None")

        """ hard curriculum dataset """
        if config.dataset == 'shd':
            train_loader, valid_loader = SHD_dataloaders(config)
            test_loader = None
        elif config.dataset == 'ssc':
            teacher_train_loader, train_loader, valid_loader, test_loader = SSC_dataloaders_teacher_student(config, teacher_config)
        elif config.dataset == 'gsc':
            train_loader, valid_loader, test_loader = GSC_dataloaders(config)
        else:
            raise Exception(f'dataset {config.dataset} not implemented')

        config.n_hidden_neurons = hidden_dim
        config.hidden_dims = 4 * hidden_dim

        logger.info("The time_step is: {}".format(config.time_step))
        logger.info("The n_bins is: {}".format(config.n_bins))
        logger.info("The hidden_dim is: {}".format(hidden_dim))
        logger.info("The spike_mode is: {}".format(config.spike_mode))
        logger.info("The block_mode is :{}".format(config.block_type))
        logger.info("The gate_v_threshold is: {}".format(config.gate_v_threshold))
        logger.info("The teacher model's time step is {}".format(teacher_config.time_step))
        model = SpikeDrivenTransformer(config).to(device)

        """
        Teacher Model
        """
        teacher_config.n_hidden_neurons = hidden_dim
        teacher_config.hidden_dims = 4 * hidden_dim
        teacher_model = SpikeDrivenTransformer(teacher_config).to(device)
        ## 1depths
        # 最好的teacher，不需要在训练的
        # model_weights_path = '/data/wjq/SNN-delays/checkpoints/ssc/spike-temporal-former||seed=312||ssc||1depths||2ms||bins=5||lr_w=0.005||heads=16||gate_v=1.0_Best_ACC_2024-07-24_19-44-52.pt'  # depth=1
        # model_weights_path = '/data/wjq/SNN-delays/checkpoints/progressive_kd/ssc/spike-temporal-former||seed=312||ssc||1depths||5ms||bins=5||lr_w=0.005||heads=16||gate_v=1.0_Best_Loss_2024-07-26_15-06-27.pt'  # depth=1
        # model_weights_path = '/data/wjq/SNN-delays/checkpoints/progressive_kd/ssc/spike-temporal-former||seed=312||ssc||1depths||10ms||bins=5||lr_w=0.005||heads=16||gate_v=1.0_Best_Loss_2024-07-28_13-00-48.pt'


        ## 2depths
        # model_weights_path = '/data/wjq/SNN-delays/checkpoints/ssc/spike-temporal-former||seed=312||ssc||2depths||2ms||bins=5||lr_w=0.005||heads=16||gate_v=1.0_Best_Loss_2024-07-24_14-48-14.pt'  # depth=2
        # model_weights_path = '/data/wjq/SNN-delays/checkpoints/progressive_kd/ssc/spike-temporal-former||seed=312||ssc||2depths||5ms||bins=5||lr_w=0.005||heads=16||gate_v=1.0_Best_ACC_2024-07-27_13-16-12.pt'  # depth=2
        model_weights_path = '/data/wjq/SNN-delays/checkpoints/progressive_kd/ssc/spike-temporal-former||seed=312||ssc||2depths||10ms||bins=5||lr_w=0.005||heads=16||gate_v=1.0_Best_Loss_2024-07-29_14-38-49.pt'  # depth=2


        # Check if the model weights file exists
        if os.path.isfile(model_weights_path):
            # Load the weights into the model
            # teacher_model.load_state_dict(torch.load(model_weights_path), strict=True)
            teacher_model.load_state_dict(torch.load(model_weights_path, map_location=device))
            logger.info(f"Teahcer Model weights loaded successfully from {model_weights_path}")
        else:
            logger.info(f"No model weights found at {model_weights_path}, starting from scratch.")

        # 讲一层的网络的权重load进去之后再次知识蒸馏
        if os.path.isfile(model_weights_path):
            # Load the weights into the model
            # teacher_model.load_state_dict(torch.load(model_weights_path), strict=True)
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            logger.info(f"Model weights loaded successfully from {model_weights_path}")
        else:
            logger.info(f"No model weights found at {model_weights_path}, starting from scratch.")

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

        schedulers = {
            "warmup": None,
            "scheduler": None
        }

        iters = len(train_loader) * config.n_warmup
        if config.n_warmup:
            schedulers["warmup"] = WarmUpLR(config, optimizer, total_iters=iters)

        logger.info("Warm_up:{}".format(config.n_warmup))

        total_iters = len(train_loader) * max(1, (teacher_config.epochs - config.n_warmup))

        # schedulers["scheduler"] = get_scheduler(optimizer, total_iters) # T_max 为剩下的迭代步长
        schedulers["scheduler"] = get_scheduler(optimizer, T)  # T_max 为config.t_max

        train_model(config, train_loader, teacher_train_loader, valid_loader, test_loader, device, model, teacher_model, optimizer, schedulers, epochs)