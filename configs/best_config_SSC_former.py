from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime


class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'ssc'
    datasets_path = '../Datasets/SSC'
    log_dir = './logs/logging/ssc/5seed'
    log_dir_test = './logs/logging/testing'

    seed = 312  # 312 42 3407 0 10086 114514+-5 3112
    seed_list = [42, 3407, 3112, 10086]
    # seed_list = [312]

    gpu = 1
    model_type = 'spikescr'
    block_type ='spikescr'  # 'sequeezeformer'

    distribute = False

    spike_mode = "lif"
    # time_step: 1,2,5,10s
    # Learning delays: time_step = 10, n_bins= 5
    depths = 2
    time_step = 10 # 100 time steps, time_step_list = [100,50,25,20,5,4]
    time_step_list = [4] # time steps: 10=> 20=> 40=>50=>100=> 200=> 250
    n_bins = 5 # 正常是5 => 700//5 = 140; 10 => 700//10 = 70
    n_warmup  = 10
    epochs = 300

    batch_size = 256
    # dropout_l control the first layer
    dropout_l = 0.1
    # dropout_p control the layers in attention
    dropout_p = 0.1
    # MLP_RATIO
    mlp_ratio = 4
    #
    split_ratio = 1

    ############################
    #        USE Module        #
    ############################
    # 控制不同模块间是否使用BN
    use_norm = False
    use_bn = True
    # 是否使用CutMix数据增强
    use_aug = True
    # 是否使用dropout
    use_dp = True
    # 是否使用DW的bias
    use_dw_bias = False

    use_global = False
    use_local = True


    ############################
    #          Augment         #
    ############################
    #    TimeNeurons_mask Aug  #
    TN_mask_aug_proba = 0.5   # 0.15
    time_mask_proportion = 0.1
    # time_mask_size = 10   # 10
    neuron_mask_size= 10  # 10

    #      CutMix Aug           #
    cutmix_aug_proba = 0.1
    cut_size_proba = 0.1

    #      TimeJitter Aug       #
    time_jitter_proba = 0.05
    max_jitter = 5

    #     NeuronJitter Aug      #
    channel_jitter_proba = 0.05

    #       DropEvent Aug       #
    drop_event_proba = 0.05
    max_drop_events = 10

    #         Noise Aug         #
    noise_proba = 0.1
    sig = 1.5

    backend = 'cupy'
    attn_mode = 'v2'
    kernel_size = 31  
    bias = True

    # weight_decay = 1e-5
    t_max = 40
    lr_w = 0.005
    weight_decay = 0.01 # Default 1e-2 => 5e-4
    n_inputs = 700//n_bins
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
    gate_v_threshold = 1.0  # LIF
    alpha = 5.0
    # surrogate_function = surrogate.Sigmoid(alpha=alpha)
    surrogate_function = surrogate.ATan(alpha=alpha)  # FastSigmoid(alpha)
    detach_reset = True
    init_w_method = 'kaiming_uniform'
    max_len = 126
    use_padding = False
    norm_type = "bn"

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
    run_info = f'||{dataset}||{depths}depths||{time_step}ms||bins={n_bins}||lr_w={lr_w}||heads={num_heads}||gate_v={gate_v_threshold}'
    wandb_run_name = run_name + f'||seed={seed}' + run_info
    # # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL_{current_time}.pt'
    make_plot = False
