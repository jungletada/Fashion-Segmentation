import models
from utils.test_model import test_mode
from dataloaders.datasets.ModanetSet import Fashionset
from dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
import utils.utils_logger as L

import os
import cv2
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm


def args_pharse():
    parser = argparse.ArgumentParser(description="Semantic Fashion Transformer")
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='True to use cuda')
    parser.add_argument('--ckpt', type=int, default=120, help='load checkpoint')
    parser.add_argument('--name', type=str, default='deeplab', help='model name')
    parser.add_argument('--show', type=bool, default=False, help='show the logs')
    args = parser.parse_args()
    return args


def show_pred(image, pred, batchsize, index, modelname, root):
    for j in range(batchsize):
        pic_index = int(j + batchsize * index)
        tmp = np.array(pred[j]).astype(np.uint8)
        seg_pr = decode_segmap(tmp, dataset='FashionSeg')
        seg_pr *= 255.0
        seg_pr = seg_pr.astype(np.uint8)

        img_tmp = np.transpose(image[j], axes=[1, 2, 0])
        img_tmp *= (0.229, 0.224, 0.225)
        img_tmp += (0.485, 0.456, 0.406)
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)

        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)
        seg_pr = cv2.cvtColor(seg_pr, cv2.COLOR_RGB2BGR)
        # save images, ground-truth, and predict segmap
        cv2.imwrite(root + "{}_input.jpg".format(pic_index), img_tmp)
        cv2.imwrite(root + "{}-{}.jpg".format(pic_index, modelname), seg_pr)


def test_model():
    args = args_pharse()
    save_models_dir = './model_zoo'
    dataset = 'ModaNet'
    net_dict = models.build_model(modelname=args.name)
    net = net_dict['net']
    modelname = net_dict['modelname']
    test_set = Fashionset(img_dir='./data/Pexels')

    save_logs_dir = './logs'
    logger_name = '__{}_{}_{}__'.format(modelname, args.ckpt, dataset)
    L.logger_info(modelname)
    L.logger_info(logger_name, log_path=os.path.join(save_logs_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info("{}_{}".format(modelname, dataset))

    root = './results_pexels/' + modelname + '/'
    if not os.path.exists(root):
        os.makedirs(root)

    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False)
    steps = len(test_loader)
    print("Steps: {}, batch size: {}".format(steps, args.batchsize))
    state = torch.load(os.path.join(
        save_models_dir, 'model_{}_{}_{}.pth'.format(modelname, dataset, args.ckpt)))
    net.load_state_dict(state)

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Total parameters: {:.2f}".format(params / 1e6))
    logger.info("Total images: {}.".format(len(test_set)))

    if args.cuda:
        net = net.cuda("cuda:0")
    net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for index, sample in enumerate(tbar):
            image = sample['image'].cuda("cuda:0")
            output = test_mode(net, image, mode=2, refield=224, min_size=224, sf=1, modulo=1)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            img = image.cpu().numpy()
            show_pred(img, pred, args.batchsize, index, modelname, root)


if __name__ == '__main__':
    test_model()
