from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime


class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'gsc'
    datasets_path = '../Datasets/GSC'
    log_dir = './logs/logging/gsc/SNN_backbone_test'
    log_dir_test = './logs/logging/testing'

    seed = 312  # 312 42 3407 0 10086 114514+-5 3112

    gpu = 3
    model_type = 'spike-temporal-former'
    block_type = 'sequeezeformer'  # 'sequeezeformer'

    distribute = False

    spike_mode = "lif"

    # Spectrogram
    # Number of Frames= 1+ (Total Length of the Audio - Window Size) / Hop_Length
    # Learning delays: hop_length=80; n_mels=140
    window_size = 64 # 256
    
    hop_length = 80  
    n_mels = 140 # 140 => 70

    depths = 2
    epochs = 300 if spike_mode == "plif" else 300
    batch_size = 64

    model_type = 'SDSA'
    block_type = 'spikformer' # 'sequeezeformer or conformer'

    # dropout_l control the first layer
    dropout_l = 0.1
    # dropout_p control the layers in attention
    dropout_p = 0.1

    # MLP_RATIO
    mlp_ratio = 4
    # SPLIT_RATIO
    split_ratio = 1
    ############################
    #        USE Module        #
    ############################
    # 控制不同模块间是否使用BN
    use_norm = False

    
    use_bn = True
    # 是否使用SepcAug数据增强
    use_aug = True
    # 是否使用dropout
    use_dp = True
    # 是否使用DW的biass
    use_dw_bias = False

    use_global = True
    use_local = False

    ############################
    #          Augment         #
    ############################

    #  SpecAugment #
    mF = 1
    F = 10
    mT = 1
    pS = 0.25




    backend = 'cupy'
    attn_mode = 'v2'
    kernel_size = 31  
    bias = True

    # weight_decay = 1e-5
    n_warmup = 0
    # lr_start = 1e-5
    t_max = 40
    lr_w = 2e-3 # 2e-3
    weight_decay = 5e-3 # Default 0.1 => 0.01 => 2e-3 => 5e-3

    n_inputs = n_mels
    n_hidden_neurons_list = [256] # [128, 144, 160, 176, 192, 208, 224, 240, 256]
    n_hidden_neurons = 144
    n_outputs = 20 if dataset == 'shd' else 35
    hidden_dims = mlp_ratio*n_hidden_neurons # 可以试一下768

    num_heads = 16  # 4=> 8=> 16 不增添加网络参数
    # spike_mode_list = ['lif', 'plif']

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss' # 'SmoothCEloss', 'CEloss'

    init_tau = 2.0 if spike_mode == "plif" else 2.0  # LIF
    v_threshold = 1.0  # LIF
    v_reset = 0.5
    gate_v_threshold = 1.0 # LIF
    alpha = 5.0

    # surrogate_function = surrogate.Sigmoid(alpha=alpha)
    surrogate_function = surrogate.ATan(alpha=alpha)  # FastSigmoid(alpha)
    detach_reset = True


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
    run_info = f'||{dataset}||{depths}depths||{window_size}ms||bins={hop_length}||lr_w={lr_w}||heads={num_heads}'
    wandb_run_name = run_name + f'||seed={seed}' + run_info
    # # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL_{current_time}.pt'
    make_plot = False
