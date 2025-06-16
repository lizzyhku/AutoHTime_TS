import argparse
import os
import random
import sys
import numpy as np
import torch
from exp.exp_main import Exp_Main

def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Timerl with Autoregressive Time Series Forecasting"
    )

    # Configuration groups
    basic = parser.add_argument_group('Basic Configuration')
    data = parser.add_argument_group('Data Configuration')
    model = parser.add_argument_group('Model Architecture')
    time_series = parser.add_argument_group('Time Series Configuration')
    training = parser.add_argument_group('Training Configuration')
    attention = parser.add_argument_group('Attention Mechanism')
    patch = parser.add_argument_group('Patch Configuration')

    # Basic Config
    basic.add_argument("--is_training", type=int, required=True, default=1,
                     help="1 for training, 0 for testing")
    basic.add_argument("--model_id", type=str, required=True,
                     default="ETTm1", help="Model identifier")
    basic.add_argument("--model", type=str, required=True,
                     default="Timerl", help="Model architecture")
    basic.add_argument("--random_seed", type=int, default=2021,
                     help="Random seed")
    basic.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    basic.add_argument("--use_multi_gpu", type=str, default=False, help="True or False")

    # Data Configuration
    data.add_argument("--data", type=str, required=True,
                     default="custom", help="Dataset type")
    data.add_argument("--root_path", type=str, default="./datasets/",
                     help="Root directory of data files")
    data.add_argument("--data_path", type=str, default="ETTm1.csv",
                     help="Data file name")
    data.add_argument("--features", type=str, default="M",
                     choices=['M', 'S', 'MS'],
                     help="Input features configuration")
    data.add_argument("--input_dim", type=int, default=7,
                     help="Input feature dimension")
    data.add_argument("--output_dim", type=int, default=7,
                     help="Output dimension")

    # Model Architecture
    model.add_argument("--hidden_dim", type=int, default=64,
                     help="Hidden dimension size")
    model.add_argument("--enc_in", type=int, default=7,
                     help="Encoder input size")
    model.add_argument("--e_layers", type=int, default=3,
                     help="Number of encoder layers")
    model.add_argument("--n_heads", type=int, default=16,
                     help="Number of attention heads")
    model.add_argument("--d_model", type=int, default=128,
                     help="Model dimension")
    model.add_argument("--d_ff", type=int, default=256,
                     help="Feedforward dimension")
    model.add_argument("--dropout", type=float, default=0.2,
                     help="Dropout rate")

    # Time Series Configuration
    time_series.add_argument("--seq_len", type=int, default=336,
                           help="Input sequence length")
    time_series.add_argument("--label_len", type=int, default=48,
                           help="Label length")
    time_series.add_argument("--pred_len", type=int, default=192,
                           help="Single-step prediction length")
    # time_series.add_argument("--total_pred_len", type=int, default=192,
    #                        help="Total prediction length (autoregressive)")
    time_series.add_argument("--chunk_size", type=int, default=48,
                           help="Prediction window size (autoregressive)")

    # PatchTST
    parser.add_argument(
        "--fc_dropout", type=float, default=0.05, help="fully connected dropout"
    )
    parser.add_argument("--head_dropout", type=float, default=0.0, help="head dropout")
    parser.add_argument(
    "--individual", type=int, default=0, help="individual head; True 1 False 0")
    parser.add_argument(
        "--padding_patch", default="end", help="None: None; end: padding on the end"
    )
    parser.add_argument("--revin", type=int, default=1, help="RevIN; True 1 False 0")
    parser.add_argument(
        "--affine", type=int, default=0, help="RevIN-affine; True 1 False 0"
    )
    parser.add_argument(
        "--subtract_last",
        type=int,
        default=0,
        help="0: subtract mean; 1: subtract last",
    )
    parser.add_argument(
        "--decomposition", type=int, default=0, help="decomposition; True 1 False 0"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=25, help="decomposition-kernel"
    )

    # Former
    parser.add_argument(
        "--embed_type",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--is_sequential", type=str, default=True, help="activation")
    
    # Training Configuration
    training.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    training.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    training.add_argument("--train_epochs", type=int, default=100,
                        help="Training epochs")
    training.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    training.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    training.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate")
    training.add_argument("--itr", type=int, default=1,
                        help="Number of experiment repetitions")
    training.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision")

    # Attention Mechanism
    attention.add_argument("--window_size", type=int, default=8,
                         help="Attention window size")
    attention.add_argument("--gamma", type=float, default=0.5,
                         help="Attention decay rate")
    attention.add_argument("--d_k", type=int, default=128,
                         help="Dimension of keys")
    attention.add_argument("--d_v", type=int, default=128,
                         help="Dimension of values")
    attention.add_argument("--alpha", type=float, default=0.1,
                         help="Attention weight parameter")
    attention.add_argument("--beta", type=float, default=0.1,
                         help="Attention bias parameter")
    attention.add_argument("--attn_decay_type", type=str, default="powerLaw",
                         choices=['powerLaw', 'exponential', 'linear'],
                         help="Type of attention decay")
    attention.add_argument("--attn_decay_scale", type=float, default=0.5,
                         help="Scale factor for attention decay")
    # FEDFormer Configuration
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')

    # Patch Configuration
    patch.add_argument("--patch_len", type=int, default=16,
                     help="Length of patches")
    patch.add_argument("--stride", type=int, default=8,
                     help="Stride for patch creation")

    
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--lradj", type=str, default="type3", help="adjust learning rate"
    )
    attention.add_argument(
        "--train_attn_decay", action="store_true",
        help="Whether to train attention decay parameters"
    )
    
    # iTransformer specific parameters
    parser.add_argument("--use_norm", type=int, default=1,
                       help="whether to use normalization in iTransformer; True 1 False 0")
    parser.add_argument("--class_strategy", type=str, default="projection",
                       help="projection strategy in iTransformer, options:[projection]")
    
    # Flowformer specific parameters
    parser.add_argument("--channel_independence", type=int, default=0,
                       help="whether to use channel independence in Flowformer; True 1 False 0")
    
    # TimeMIxer
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    args = parser.parse_args()
    
    # parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--window_size', type=int, default=32)
    # parser.add_argument('--patch_len', type=int, default=16)
    # parser.add_argument('--gamma', type=float, default=0.5)
        
    # Post-processing
    args.use_gpu = True if torch.cuda.is_available() else False
    if args.use_gpu:
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')
    
    return args

def setup_environment(args):
    """Configure runtime environment"""
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

def generate_experiment_name(args, iteration):
    """Generate consistent experiment naming"""
    components = [
        args.model_id,
        args.model,
        args.data,
        f"in{args.input_dim}",
        f"out{args.output_dim}",
        f"hid{args.hidden_dim}",
        f"ft{args.features}",
        f"sl{args.seq_len}",
        f"ll{args.label_len}",
        f"pl{args.pred_len}",
        # f"tpl{args.total_pred_len}",
        f"cs{args.chunk_size}",
        f"dm{args.d_model}",
        f"nh{args.n_heads}",
        f"el{args.e_layers}",
        f"ws{args.window_size}",
        f"g{args.gamma}",
        f"dk{args.d_k}",
        f"dv{args.d_v}",
        f"a{args.alpha}",
        f"b{args.beta}",
        f"adt{args.attn_decay_type[:4]}",
        f"ads{args.attn_decay_scale}",
        f"pt{args.patch_len}",
        f"st{args.stride}",
        f"lr{args.learning_rate}",
        f"bs{args.batch_size}",
        f"itr{iteration}"
    ]
    return "_".join(components)

def main():
    print(torch.cuda.device_count())
    args = parse_arguments()
    
    setup_environment(args)
    
    Exp = Exp_Main
    
    if args.is_training:
        for iteration in range(args.itr):
            setting = generate_experiment_name(args, iteration)
            print(f">>>>>>> Starting Experiment: {setting} <<<<<<<")
            
            if os.path.exists("./result.txt"):
                with open("./result.txt", "r") as f:
                    if any(setting in line for line in f):
                        print(f"Results exist for {setting}")
                        continue
            
            exp = Exp(args)
            
            print(">>>>>>> Training Phase <<<<<<<")
            exp.train(setting)
            
            print(">>>>>>> Testing Phase <<<<<<<")
            exp.test(setting, test=1, autoregressive=True)
            
            torch.cuda.empty_cache()
    else:
        setting = generate_experiment_name(args, args.itr)
        exp = Exp(args)
        print(f">>>>>>> Testing Only: {setting} <<<<<<<")
        exp.test(setting, test=1, autoregressive=True)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()