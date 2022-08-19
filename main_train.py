from dataloaders.datasets.SegmentationSet import SegmentationDataset
from dataloaders.datasets.ModanetSet import ModaSegmentation
from utils.lr_scheduler import create_lr_scheduler, create_poly_lr_scheduler
from utils.semantic_loss import build_critierion, get_loss_aux, get_loss_NO_aux
from torch.utils.data import DataLoader
import utils.utils_logger as L
import models
import os
import torch
import logging
import argparse


def args_pharse():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Training")
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='True to use cuda')
    parser.add_argument('--epochs', type=int, default=130, help='number of epochs')
    parser.add_argument('--ckpt', type=int, default=0, help='load checkpoint')
    parser.add_argument('--name', type=str, default='fashion_s', help='model name')
    parser.add_argument('--dataset', type=str, default='ModaNet', help='choose the dataset')
    args = parser.parse_args()
    return args


def train_model():
    args = args_pharse()
    dataset = args.dataset
    save_models_dir = './model_zoo'
    save_logs_dir = './logs'
    lambda_ = 0.4
    # build model and logger
    net_dict = models.build_model(modelname=args.name)
    net = net_dict['net']
    modelname = net_dict['modelname']

    lr = 1.25e-3 * (args.batchsize / 32)
    optimizer = torch.optim.AdamW(
        [{'params': [p for p in net.parameters() if p.requires_grad],
          'initial_lr': lr}],
        lr=lr, betas=(0.9, 0.99),
        weight_decay=5e-2)

    if args.cuda:
        net = net.cuda()
        torch.cuda.empty_cache()

    if args.ckpt != 0:
        net.load_state_dict(torch.load(
            os.path.join(save_models_dir,'model_{}_{}_{}.pth'.format(modelname, dataset, args.ckpt))))
        optimizer.load_state_dict(torch.load(
            os.path.join(save_models_dir,'optim_{}_{}_{}.pth'.format(modelname, dataset, args.ckpt))))


    logger_name = 'logger_{}_{}'.format(modelname, dataset)
    L.logger_info(logger_name, log_path=os.path.join(save_logs_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Total parameters: {:.2f}".format(params / 1e6))

    if dataset == 'ModaNet':
        train_set = ModaSegmentation(
            img_dir='./data/ModaNet/train/image',
            anno_path='data/annotations',
            json_file='ModaNet_train.json',
            id_pth='ModaNet_train_ids.pth',
            phase='train')

    elif dataset == 'DeepFashion2':
        train_set = SegmentationDataset(
            img_dir='./data/DeepFashion2/train/image',
            anno_path='data/annotations',
            json_file='DeepFashion2_train.json',
            id_pth='DeepFashion2_train_ids.pth',
            phase='train'
        )

    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    steps = len(train_loader)
    scheduler = create_poly_lr_scheduler(optimizer,
                                    steps,
                                    args.epochs,
                                    warmup=True,
                                    warmup_epochs=1,
                                    end_warmup_epochs=10,
                                    power=0.9,
                                    last_epoch=args.ckpt)

    begin = args.ckpt + 1
    end = args.epochs + 1
    criterion = build_critierion(cuda=args.cuda, mode='focal')
    for epoch in range(begin, end):
        running_loss = 0
        for i, sample in enumerate(train_loader):
            image = sample['image'].cuda()
            target = sample['label'].cuda()
            optimizer.zero_grad()
            out_dict = net.forward(image)
            loss1, loss2, loss = get_loss_aux(criterion, out_dict, target, weight=lambda_)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss1.item()

            if i % 50 == 0:
                logger.info("Epoch: [{}/{}], Step: [{}/{}], Loss1: {:.5f}, Loss2:{:.5f}".
                            format(epoch, end - 1, i, steps, loss1.item(), loss2.item()))

        logger.info("epoch {}, loss1:{:.5f}".format(epoch, running_loss/steps))

        if epoch % 10 == 0 and epoch != 0:
            torch.save(net.state_dict(),
                       os.path.join(save_models_dir, 'model_{}_{}_{}.pth'.format(modelname, dataset, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(save_models_dir, 'optim_{}_{}_{}.pth'.format(modelname, dataset, epoch)))


if __name__ == '__main__':
    train_model()
