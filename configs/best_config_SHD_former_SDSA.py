from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime


class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'shd'
    datasets_path = '../Datasets/SHD'
    log_dir = './logs/logging/shd/SNN_backbone_test'
    log_dir_test = './logs/logging/testing'

    seed = 312 # 312 42 3407 0 10086 114514+-5 3112
    gpu = 0

    depths = 1
    distribute = False

    spike_mode = "lif" # "lif", "plif", "ttfs"
    # time_step: 1,2,5,10,25
    time_step = 10 # 10 time steps
    n_bins = 1 # 700 neuron
    epochs = 300
    n_warmup = 10
    batch_size = 16 # 128 => 256 = > 512
    use_tim = False
    model_type = 'SDSA'
    block_type = 'spikformer' # 'sequeezeformer or conformer'
    ############################
    #        USE Module        #
    ############################
    # 控制不同模块间是否使用BN
    use_norm = False


    
    use_bn = True
    # 是否使用Aug数据增强
    use_aug = True
    # 是否使用dropout
    use_dp = True
    # 是否使用DW的bias
    use_dw_bias = False




    ############################
    #          Augment         #
    ############################
    #    TimeNeurons_mask Aug  #
    TN_mask_aug_proba = 0.5
    # time_mask_size = 5
    time_mask_proportion = 0.2
    neuron_mask_size= 20

    #      CutMix Aug           #
    cutmix_aug_proba = 0.1
    cut_size_proba = 0.4

    #      TimeJitter Aug       #
    time_jitter_proba = 0.05
    max_jitter = 1

    #     NeuronJitter Aug      #
    channel_jitter_proba = 0.05

    #       DropEvent Aug       #
    drop_event_proba = 0.05
    max_drop_events = 5

    #         Noise Aug         #
    noise_proba = 0.1
    sig = 0.75


    loss_fn = 'CEloss'

    backend = 'cupy'
    n_outputs = 20 if dataset == 'shd' else 35


    t_max = 40 # 40
    lr_w = 5e-3 # 1e-3
    weight_decay = 0.01  # Default 1e-2 => 5e-3 => 1e-4

    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adamw'
    optimizer_pos = 'adamw'


    ################################################
    #                    Save                      #
    ################################################
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = model_type
    run_info = f'||{dataset}||{model_type}||{depths}depths||{time_step}ms||bins={n_bins}||lr_w={lr_w}||{model_type}'
    wandb_run_name = run_name + f'||seed={seed}' + run_info
    # # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL_{current_time}.pt'
    make_plot = False
