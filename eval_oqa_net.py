import os
import torch
import argparse
import json

from imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from imagenet_codebase.run_manager import ImagenetRunConfig, TestRunManager
from model_zoo import ofa_net
from elastic_nn.modules.lsq_conv import LsqConv
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    help='The path of imagenet',
    type=str,
    default='/dataset/imagenet')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=100)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=4)
parser.add_argument(
    '-is',
    '--image_size',
    help='size of active image',
    type=int,
    default=224)
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    default='ofa_mbv3_d234_e346_k357_w1.0',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'oqa_mbv3_d234_e346_k357_w1.0_b4', 'oqa_mbv3_d234_e346_k357_w1.0_b3','oqa_mbv3_d234_e346_k357_w1.0_b2', 'oqa_mbv3_d234_e346_k357_w1.0_b1'],
    help='OFA networks')
parser.add_argument(
    '-ks',
    '--ks_list',
    metavar='kernel options for all blocks',
    default='7')
parser.add_argument(
    '-es',
    '--expand_list',
    metavar='width options for all blocks',
    default='6')
parser.add_argument(
    '-ds',
    '--depth_list',
    metavar='depth options for all blocks',
    default='4')
args = parser.parse_args()
args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
args.expand_list = [int(e) for e in args.expand_list.split(',')]
args.depth_list = [int(d) for d in args.depth_list.split(',')]
if len(args.ks_list) == 1:
    args.ks_list = args.ks_list[0]
if len(args.expand_list) == 1:
    args.expand_list = args.expand_list[0]
if len(args.depth_list) == 1:
    args.depth_list = args.depth_list[0]

if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
#ImagenetDataProvider.DEFAULT_PATH = args.path

ofa_network,_ = ofa_net(args.net, pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
# ofa_network.set_active_subnet(ks=7, e=6, d=4)
# subnet = ofa_network.get_active_subnet(preserve_weight=True)
from datetime import datetime

if 'oqa' in args.net:
    save_path = 'quant_exp/calculate_flops'
else:
    save_path = 'exp/calculate_flops'
if not os.path.exists(save_path):
    print("env make dir: " + save_path)
    try:
        os.makedirs(save_path)
    except Exception as e:
        print(e)
        pass
ofa_network.set_active_subnet(ks=args.ks_list, e=args.expand_list, d=args.depth_list)
subnet = ofa_network.get_active_subnet(preserve_weight=True)
""" Test sampled subnet 
"""
run_manager = TestRunManager(save_path, subnet, run_config, init=False)
# assign image size: 128, 132, ..., 224
run_config.data_provider.assign_active_img_size(args.image_size)
run_manager.reset_running_statistics(net=subnet)

from imagenet_codebase.utils.pytorch_utils import count_net_flops, count_parameters
fp_flops = count_net_flops(subnet,(1,3,args.image_size,args.image_size)) / 1e6
fp_params = count_parameters(subnet) / 1e6
print('fp_flops: ',fp_flops, ' fp_params: ', fp_params)

from imagenet_codebase.utils.flops_counter import profile_quant
flops,params = profile_quant(subnet,(1,3,args.image_size,args.image_size))
print('flops: ',flops, ' params: ', params)

print('Test random subnet:')
print(subnet.module_str)

loss, top1, top5, newsubnet = run_manager.validate(net=subnet)
print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

filepath = save_path+'/net_info.txt'
with open(filepath, 'a') as fout:
    fout.write('load model from: '+args.net+'\n')
    fout.write(subnet.module_str+'\n')
    fout.write('kernel setting: '+json.dumps(args.ks_list)+'\n')
    fout.write('expand setting: '+json.dumps(args.expand_list)+'\n')
    fout.write('depth setting: '+json.dumps(args.depth_list)+'\n')
    val_log = '#Subnet with ImageSize {5}\tFLOPs {0:.3f}M\tParams {1:.3f}M\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\n'.\
                    format(flops, params, loss, top1, top5, args.image_size)

    print(val_log)
    fout.write(val_log + '\n')
    fout.write('\n')
    #fout.flush()
    fout.close()
