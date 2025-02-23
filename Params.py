import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--save_path', default='temp', help='file name to save model and training record')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size = latdim // 2')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
    parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
    parser.add_argument('--normalization', default='left', type=str, help='type of normalization')
    parser.add_argument('--encoder', default='lightgcn', type=str, help='type of encoder')
    parser.add_argument('--num_exps', default=3, type=int, help='number of expert')
    parser.add_argument('--stop_gradient', default=True, type=bool, help='sg')
    parser.add_argument('--gnn_layer', default='[1,1,1]', help='number of gnn layers')
    parser.add_argument('--gnn_mtl_layer', default='[1,1,1]', help='number of gnn_mtl layers')
    parser.add_argument('--coefficient', default='[2.0/6, 2.0/6, 2.0/6]' , help='hyperparam for each beh')
    parser.add_argument('--data', default='beibei', type=str, help='name of dataset')
    parser.add_argument('--mult', default=300, type=float, help='multiplier for the result')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--decoder', default='dfme', type=str, help='type of decoder')
    parser.add_argument('--gnn', default='cogcn', type=str, help='type of multi behavior')
    parser.add_argument('--src', default=0.001, type=float, help='hyperparam for src')
    parser.add_argument('--index', default=0.1, type=float, help='hyperparam for index')
    parser.add_argument('--cl', default=1, type=int, help='hyperparam for cl')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--ssl_temp', default=0.1, type=float, help='hyperparam for temp')
    parser.add_argument('--ssl_reg', default=1.0, type=float, help='hyperparam for reg')
    parser.add_argument('--ssl_mode', default='both_side', type=str, help='user_side item_side both_side')

    return parser.parse_args()
args = parse_args()

if args.data == 'beibei':
    args.trnNum = 21716
    args.coefficient = "[0.0/6, 2.0/6, 4.0/6]"
    print('setting for beibei')
elif args.data == 'taobao':
    args.trnNum = 15449
    args.coefficient = "[2.0/6, 2.0/6, 2.0/6]"
    print('setting for taobao')
elif args.data == 'tmall':
    args.trnNum = 41738
    args.coefficient = "[1.0/6, 2.0/6, 0.0/6, 3.0/6]"
    print('setting for tmall')

args.decay_step = args.trnNum // args.batch
