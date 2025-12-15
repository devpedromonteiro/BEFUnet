import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob

from models.BEFUnet import BEFUnet
import configs.BEFUnet_configs as configs
from trainer import trainer


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=401, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=10, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=2,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='BEFUnet')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--resume', type=str,
                    default=None, help='path to checkpoint to resume from (if None, will auto-detect latest checkpoint)')
parser.add_argument('--weight_ce', type=float,
                    default=0.4, help='weight for Cross Entropy loss')
parser.add_argument('--weight_dice', type=float,
                    default=0.5, help='weight for Dice loss')
parser.add_argument('--weight_boundary', type=float,
                    default=0.1, help='weight for Boundary loss')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    CONFIGS = {
        'BEFUnet': configs.get_BEFUnet_configs(),
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


    model = BEFUnet(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    
    # Handle resume checkpoint
    resume_path = args.resume
    if resume_path is None:
        # Auto-detect latest checkpoint if exists
        checkpoint_pattern = os.path.join(args.output_dir, f'{args.model_name}_checkpoint_epoch_*.pth')
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            # Sort by epoch number (extract from filename)
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            resume_path = checkpoints[-1]  # Get the latest checkpoint
            print(f"Auto-detected latest checkpoint: {resume_path}")
        else:
            print("No checkpoint found. Starting training from scratch.")
    
    trainer(args, model, args.output_dir, resume_path=resume_path)
