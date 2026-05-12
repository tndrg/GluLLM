import argparse
def args_bglp():
    parser = argparse.ArgumentParser(description='GluLLM')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, in_context_forecasting]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--mn', type=str, default=None, help='model name')
    parser.add_argument('--baseline',action='store_true', help='whether to use baseline methods')
    parser.add_argument('--test_model',action='store_true', help='whether to test model only')
    parser.add_argument('--results_dir',type=str, default='test_results', help='where to save testing results')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test'], help='Execution mode')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')

    # data loader
    parser.add_argument('--data', type=str, default='glucose', help='dataset type')
    parser.add_argument('--ds', type=str, default=None, help='dataset name')
    parser.add_argument('--pid', type=str, default=None, help='subject ID')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last',action='store_true', help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle',action='store_true', help='shuffle validation set')
    parser.add_argument('--population', action='store_false', help='whether to use population setting')
    parser.add_argument('--data_base', default='.', help='data base to load data')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=72, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=60, help='label length')
    parser.add_argument('--token_len', type=int, default=12, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=72, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=60, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=12, help='test pred len')


    # model define
    parser.add_argument('--cache_dir', type=str, default='.', help='location of HF LLM cache')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default=None, help='llm checkpoints dir')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')

    # baseline define
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder (baseline)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model (baseline)')
    parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')       
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method, only support avg, max, conv')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp',action='store_false',help='use automatic mixed precision training (default true)')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--mix_embeds', help='mix embeds', action='store_false')
    parser.add_argument('--use_prompt', help='whether to use personalised prompts', action='store_false')
    parser.add_argument('--use_scheduler', action='store_true',  help='Use learning rate scheduler')

    # GPU
    # parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', help='use multiple gpus', action='store_true')
    parser.add_argument('--visualize', action='store_true', help='visualize')
    args = parser.parse_args()

    print(args)
    return args
