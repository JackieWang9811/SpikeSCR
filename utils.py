import numpy as np
import random, sys
import torch
import logging
import os
import datetime
from logging import getLogger
import yaml
import re
from torch import optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_config_dict(config_dict):
    """This function convert the str parameters to their original type.
    """
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                value = param
        except (NameError, SyntaxError, TypeError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def read_configuration(config_file):
    # read configuration from yaml file
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        yaml_config_dict = yaml.load(f.read(), Loader=yaml_loader)

    # read configuration from cmd line
    cmd_config_dict = dict()
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        # logger.warning('command line args [{}] will not be used in TextBox'.format(' '.join(unrecognized_args)))

    cmd_config_dict = convert_config_dict(cmd_config_dict)

    final_config_dict = dict()
    final_config_dict.update(yaml_config_dict)
    final_config_dict.update(cmd_config_dict)

    return final_config_dict



def check_versions():
    python_version = sys.version .split(' ')[0]
    print("============== Checking Packages versions ================")
    print(f"python {python_version}")
    print(f"numpy {np.__version__}")
    print(f"pytorch {torch.__version__}")



def set_seed(seed):
    print(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # print("Setting seed successfully!")


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def init_logger(config, type):

    if type == "training":

        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if config.dataset == "gsc" or config.dataset == "gsc12":
            if config.use_aug:
                logfilename = '{}-{}-seed{}-depth{}-hop_length{}-lr{}-decay{}-t_max{}-mF{}-F{}-mT{}-pS{}.log'.format(config.dataset, get_local_time(),
                                                                                                config.seed, config.depths, config.hop_length, config.lr_w, config.weight_decay, config.t_max,
                                                                                                config.mF, config.F, config.mT, config.pS)
            else:
                logfilename = '{}-{}-seed{}-depth{}-hop_length{}-lr{}-decay{}-t_max{}.log'.format(config.dataset, get_local_time(),
                                                                                                config.seed, config.depths, config.hop_length, config.lr_w, config.weight_decay, config.t_max)
        else:
            if config.use_aug:
                logfilename = '{}-{}-seed{}-depth{}-time_step{}-lr{}-decay{}-t_max{}-proba{}-t_mask{}-n_mask{}.log'.format( config.dataset, get_local_time(),
                                                                                                       config.seed, config.depths, config.time_step, config.lr_w, config.weight_decay,config.t_max,
                                                                                                       config.TN_mask_aug_proba, config.time_mask_proportion, config.neuron_mask_size)
            else:
                logfilename = '{}-{}-seed{}-depth{}-time_step{}-lr{}-decay{}-t_max{}.log'.format( config.dataset, get_local_time(),
                                                                                                       config.seed, config.depths, config.time_step, config.lr_w, config.weight_decay,config.t_max)

        logfilepath = os.path.join(config.log_dir, logfilename)

    else:
        if not os.path.exists(config.log_dir_test):
            os.makedirs(config.log_dir_test)

        if config.dataset == "gsc" or config.dataset == "gsc12":
            logfilename = '{}-{}-{}-seed{}-depth{}-lr{}-decay{}-mF{}-F{}-mT{}-pS{}.log'.format( config.model_type, config.dataset, get_local_time(),
                                                                                                config.seed, config.depths, config.lr_w, config.weight_decay,
                                                                                                config.mF, config.F, config.mT, config.pS)
        else:
            logfilename = '{}-{}-{}-seed{}-depth{}-lr{}-decay{}-proba{}-t_mask{}-n_mask{}.log'.format( config.model_type, config.dataset, get_local_time(),
                                                                                                       config.seed, config.depths, config.lr_w, config.weight_decay,
                                                                                                       config.TN_mask_aug_proba, config.time_mask_proportion, config.neuron_mask_size)
        logfilepath = os.path.join(config.log_dir_test, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logger = getLogger()  # 创建一个名为config.model_type的logger
    logger.setLevel(level)  # 设置logger的level
    logger.addHandler(fh)  # 给logger添加file handler
    logger.addHandler(sh)  # 给logger添加stream handler

    return logger  # 返回配置好的logger实

def build_optimizer(config,model):
    ##################################
    #  returns a list of optimizers
    ##################################
    # optimizers_return = []

    if config.model_type in ['snn_delays', 'snn_delays_lr0', 'snn']:
        if config.optimizer_w == 'adam':
            optimizer = optim.Adam([{'params':config.weights, 'lr':config.lr_w, 'weight_decay':config.weight_decay},
                                                 {'params':config.weights_plif, 'lr':config.lr_w, 'weight_decay':config.weight_decay},
                                                 {'params':config.weights_bn, 'lr':config.lr_w, 'weight_decay':0}])
        if config.model_type == 'snn_delays':
            if config.optimizer_pos == 'adam':
                optimizer = optim.Adam(config.positions, lr = config.lr_pos, weight_decay=0)
    elif config.model_type in ['spike-driven-former','spike-temporal-former','ann', 'TMI', 'Spikformer','SDSA', 'QKFormer']:
        if config.optimizer_w == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr_w, betas=(0.9, 0.999))
        elif config.optimizer_w == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config.lr_w, betas=(0.9, 0.999), weight_decay=config.weight_decay) # weight_decay default 1e-2
    return optimizer