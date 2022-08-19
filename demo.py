import numpy as np
import matplotlib.pyplot as plt
import cv2

root = './results_imgs/'


def mask2img(img_mask, img_input):
    rows, cols, channels = img_mask.shape
    roi = img_input[0:rows, 0:cols]
    # 原始图像转化为灰度值
    img2gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    # 将灰度值二值化，得到ROI区域掩模
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # ROI掩模区域反向掩模
    mask_inv = cv2.bitwise_not(mask)
    # 掩模显示背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # 掩模显示前景
    img2_fg = cv2.bitwise_and(img_mask, img_mask, mask=mask)
    # 前背景图像叠加
    dst = cv2.addWeighted(img1_bg, 0.4, img2_fg, 0.9, 0)
    return dst
    # cv2.imshow('res', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def read_imgs(indices='76'):
    suffix = '.png'
    path = './results_pexels/'
    models = ['fashion_b-03-17', 'semask_b', 'shunted_b', 'swin_b',
              'SegFormer_b5', 'DANet', 'Deeplab_v3+', 'pspnet']
    name_in = f"{indices}_input.jpg"
    name_gt = f"{indices}_GT{suffix}"
    # ground truth mask
    img_in = cv2.imread(path + name_in)
    img_gt = cv2.imread(path + name_gt)
    gt_mask = mask2img(img_gt, img_in)
    result_mask = [gt_mask]

    # prediction
    for model in models:
        name_pred = f"{indices}-{model}.jpg"
        img_pred = cv2.imread(path + name_pred)
        pred_mask = mask2img(img_pred, img_in)
        result_mask.append(pred_mask)

    result_mask = np.concatenate(result_mask, axis=1)
    cv2.imwrite(root+indices+'_Results.png', result_mask)
    cv2.imshow('res', result_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    indices = ['0', '1', '2', '6', '13', '15']
    outs = []
    for index in indices:
        read_imgs(indices=index)
        outs.append(cv2.imread(root + index + '_Results.png'))

    combine_mask = np.concatenate(outs, axis=0)
    cv2.imwrite(root + 'result_pic.png', combine_mask)
    cv2.imshow('res', combine_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
