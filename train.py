import os
from tqdm import tqdm  # 进度条
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

from inits import init_net
from models import Unet, AttUnet, Unet3plus
import blocks
from dataset import PlaygroundTrainDataset
import utils
# import metrics
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device.type}:{device.index}", end='')
    utils.print_log(args)

    model_save_dir = utils.make_dir(args.save_folder)       # 检查保存模型的文件夹

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size))
    ])

    train_data = PlaygroundTrainDataset(root=args.root, mode='train', transform=trans, num_classes=args.num_classes)
    train_iter = DataLoader(train_data, batch_size=args.batch_size)
    # val_data = PlaygroundTrainDataset(root=args.root, mode='validate', transform=trans, num_classes=args.num_classes)
    # val_iter = DataLoader(train_data, batch_size=args.batch_size)

    # net = Unet(backbone='unet_backbone', in_channel=args.in_channel, num_classes=args.num_classes)
    net = AttUnet(backbone=args.backbone, attn_block=blocks.Attention_block_SE,
                  in_channel=args.in_channel, num_classes=args.num_classes)
    # net = Unet3plus(backbone='unet_backbone', in_channel=args.in_channel, num_classes=args.num_classes)
    net = init_net(net)
    net.to(device)

    # ------------------------------------------------------------------------------------------------ #
    #                               设置optimizer lr_scheduler和加载已训练模型                             #
    # ------------------------------------------------------------------------------------------------ #
    # 只去更新优化需要训练的网络参数（即需要计算梯度的网络参数）
    # params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
    #                             momentum=args.momentum, weight_decay=args.weight_decay)
    assert args.end_epoch >= args.start_epoch
    num_epoch = args.end_epoch - args.start_epoch
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.lr_scheduler == 'multistep':
        args.lr_steps = [int(item) for item in args.lr_steps]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=1)

    # 如果传入load参数, 即上次训练的权重路径, 则接着上次的参数继续训练
    if args.load:
        checkpoint = torch.load(args.load, map_location='cpu')
        # 仅加载checkpoint在net里面有的key-value
        pretrained_dict = checkpoint['net']
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)      # 2. overwrite entries in the existing state dict
        net.load_state_dict(model_dict)         # 3. load the new state dict

        if args.load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if args.load_lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        losses_log = utils.read_log(os.path.join(model_save_dir, 'loss_log.txt'))
        scores_log = utils.read_log(os.path.join(model_save_dir, 'score_log.txt'))
    else:
        losses_log = []
        scores_log = []

    # ---------------------------------------------------------------------------------------------------- #
    #                                               开始炼丹                                                #
    # ---------------------------------------------------------------------------------------------------- #
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.start_epoch, args.end_epoch):
        total_loss = 0
        train_bar = tqdm(train_iter, unit='step')
        # ------------------------------------------------------------------------------------------------ #
        #                                           training loop                                          #
        # ------------------------------------------------------------------------------------------------ #
        net.train()
        for img, mask in train_bar:
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = net(img)
                loss = loss_fn(output, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss
            lr_now = lr_scheduler.get_lr()[0]
            train_bar.set_description(f'epoch [{epoch + 1}/{args.end_epoch}]  Loss: '
                                        f'{loss.item():.10f}, lr: {lr_now}')
        aver_loss = (total_loss / len(train_bar)).item()
        loss_log = f'epoch [{epoch + 1}/{args.end_epoch}]  lr = {lr_now:.10f}  Average loss = {aver_loss}'
        # loss_log = f'epoch [{epoch + 1}/{args.end_epoch}]  Average loss = {aver_loss}'
        print(loss_log)
        losses_log.append(loss_log)
        lr_scheduler.step()

        # ------------------------------------------------------------------------------------------------ #
        #                                          validation loop                                         #
        # ------------------------------------------------------------------------------------------------ #
        # 每args.val_step个epoch, 进行一次validation计算
        if (epoch + 1) % args.val_step == 0:
            net.eval()
            pass

        if args.save:
            # 每args.save_step个epoch, 存一次模型参数和其他有的没的
            if (epoch + 1) % args.save_step == 0:
                save_files = {
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    'epoch': epoch + 1}
                torch.save(save_files, os.path.join(model_save_dir, f'model_{epoch+1}.pth'))
        utils.write_log(losses_log, log_path=os.path.join(model_save_dir, 'loss_log.txt'), describe='')
        utils.write_log(scores_log, log_path=os.path.join(model_save_dir, 'score_log.txt'), describe='')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    # 基本参数设置
    parser.add_argument('--device', default='cuda:0', type=str, help='训练使用的GPU, 默认是使用单卡训练')
    parser.add_argument('--save', action='store_true', help='是否保存模型（default: False）')
    parser.add_argument('--save_folder', default='./checkpoints', type=str, help='model saved folder')
    parser.add_argument('--root', default='../data/MICCAI_pre_test_data', help='')
    # parser.add_argument('--val_dir', default='./All_images/data/val', type=str, help='')
    parser.add_argument('--img_size', default=256, type=int, help='image size')
    parser.add_argument('--in_channel', default=1, type=int, help='image input channel')
    parser.add_argument('--num_classes', default=5, type=int, help='分割类别数（包括背景类）')
    # parser.add_argument('--augment', action='store_true', help='默认不使用数据增强')

    # 加载已训练权重选项
    parser.add_argument('--load', default='', type=str, help='path of loaded checkpoint, 使用该参数则表示加载权重')
    parser.add_argument('--load_optimizer', action='store_true', help='默认不加载optimizer')
    parser.add_argument('--load_lr_scheduler', action='store_true', help='默认不加载lr_scheduler')

    # 网络参数
    parser.add_argument('--backbone', default='resnet50', type=str, help='backbone类型')

    # 网络训练基本参数
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--end_epoch', default=20, type=int, help='end epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for SGD (default: 0.01)')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--amp', action='store_true', help='默认不使用混合精度计算')
    parser.add_argument('--val_step', default=1, type=int, help='每隔多少个epoch进行一次validation')
    parser.add_argument('--save_step', default=10, type=int, help='每隔多少个epoch保存一次模型')

    # 优化器optimizer和学习率scheduler参数
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_steps', default=[], nargs='+', help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='lr_scheduler, option：cosine, multistep, constant(其它所有的输入都会默认为constant)')

    args = parser.parse_args()

    main(args)
