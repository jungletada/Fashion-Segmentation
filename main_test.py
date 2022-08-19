from dataloaders.datasets.SegmentationSet import SegmentationDataset
from dataloaders.datasets.ModanetSet import ModaSegmentation
from torch.utils.data import DataLoader
from dataloaders.utils import decode_segmap
from utils.metrics import Evaluator
import utils.utils_logger as L
from  utils.semantic_loss import build_critierion, get_loss_NO_aux
from tqdm import tqdm
import models
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import torch
import os


def args_pharse():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Training")
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='True to use cuda')
    parser.add_argument('--ckpt', type=int, default=120, help='load checkpoint')
    parser.add_argument('--name', type=str, default='deeplab', help='model name')
    parser.add_argument('--show', type=bool, default=False, help='show the results')
    parser.add_argument('--dataset', type=str, default='ModaNet', help='choose the dataset')
    args = parser.parse_args()
    return args


def show_pred(image, target, pred, batchsize):

    for j in range(batchsize):
        tmp = np.array(target[j]).astype(np.uint8)
        seg = decode_segmap(tmp, dataset='FashionSeg')

        tmp = np.array(pred[j]).astype(np.uint8)
        seg_pr = decode_segmap(tmp, dataset='FashionSeg')

        img_tmp = np.transpose(image[j], axes=[1, 2, 0])
        img_tmp *= (0.229, 0.224, 0.225)
        img_tmp += (0.485, 0.456, 0.406)
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        plt.figure()

        plt.title('display')
        plt.subplot(131)
        plt.imshow(img_tmp)

        plt.subplot(132)
        plt.imshow(seg)

        plt.subplot(133)
        plt.imshow(seg_pr)

    plt.show(block=True)


def test_model():
    args = args_pharse()
    save_models_dir = './model_zoo'
    save_logs_dir = './logs'

    dataset = args.dataset
    net_dict = models.build_model(modelname=args.name)
    net = net_dict['net']
    modelname = net_dict['modelname']
    logger_name = '_{}_{}_{}'.format(modelname, args.ckpt, dataset)
    L.logger_info(modelname)
    L.logger_info(logger_name, log_path=os.path.join(save_logs_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info("{}_{}".format(modelname, dataset))

    critierion = build_critierion(cuda=args.cuda, mode='focal')

    if dataset == 'ModaNet':
        test_set = ModaSegmentation(
            img_dir='./data/ModaNet/train/image',
            anno_path='data/annotations',
            json_file='ModaNet_train.json',
            id_pth='ModaNet_train_ids.pth',
            phase='test')

    elif dataset == 'DeepFashion2':
        test_set = SegmentationDataset(
            img_dir='./data/DeepFashion2/validation/image',
            anno_path='data/annotations',
            json_file='DeepFashion2_validation.json',
            id_pth='DeepFashion2_validation_ids.pth',
            phase='test'
        )

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Total parameters: {:.2f}".format(params / 1e6))
    logger.info("Total images: {}.".format(len(test_set)))
    # dataloader
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True)
    steps = len(test_loader)
    logger.info("Steps: {}, batch size: {}".format(steps, args.batchsize))
    # evaluator
    evaluator = Evaluator(num_class=14,
                          save_path=save_logs_dir+'/confusion_matrix_{}_{}.npy'.format(modelname, dataset))
    evaluator.reset()

    state = torch.load(os.path.join(save_models_dir, 'model_{}_{}_{}.pth'.format(modelname, dataset, args.ckpt)))
    net.load_state_dict(state)

    if args.cuda:
        net = net.cuda()

    net.eval()
    tbar = tqdm(test_loader)
    total_loss = 0
    steps = len(test_loader)
    with torch.no_grad():
        for index, sample in enumerate(tbar):
            image = sample['image'].cuda()
            target = sample['label'].cuda()

            out_dict = net.forward(image)
            loss = get_loss_NO_aux(critierion, out_dict, target)
            total_loss += loss.item()
            pred = out_dict['output'].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()

            evaluator.add_batch(target, pred)
            tbar.set_description("Step: [{}/{}]".format(index, steps))

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU, mIoU_avg = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    # evaluator.save_matrix()
    logger.info('val/mIoU: {}'.format(mIoU_avg))
    logger.info('val/Acc: {}'.format(Acc))
    logger.info('val/Acc_class: {}'.format(Acc_class))
    logger.info('val/fwIoU: {}'.format(FWIoU))
    logger.info('val/mIoU class:')
    for cls_IoU in mIoU:
        logger.info('{}'.format(cls_IoU))
    logger.info('Total Loss: {:.5f}'.format(total_loss))
    logger.info('Average Loss: {:.7f}'.format(total_loss/steps))
    # if args.show and epoch == 0:
    #     img = image.cpu().numpy()
    #     show_pred(img, target, pred, args.batchsize)
    #     break

if __name__ == '__main__':
    test_model()