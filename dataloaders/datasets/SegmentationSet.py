import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys 
sys.path.append("../..") 
from tqdm import trange
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloaders.utils import decode_segmap

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SegmentationDataset(Dataset):
    def __init__(self, img_dir='../../data/ModaNet/train',
                 anno_path='../../annotations',
                 json_file='ModaNet_train.json',
                 id_pth = 'ModaNet_train_ids.pth',
                 phase='train'):
        super().__init__()
        self.base_size = 256
        self.crop_size = 224

        self.img_dir = img_dir
        self.anno_path = anno_path
        self.phase = phase
        self.NUM_CLASSES = 14
        # self.class_count = dict([(x, 0) for x in range(self.NUM_CLASSES)])
        ann_file = os.path.join(self.anno_path, json_file)
        ids_file = os.path.join(self.anno_path, id_pth)

        self.moda = COCO(ann_file)
        self.moda_mask = mask

        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.moda.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.phase == "train":
            return self.transform_tr(sample)

        elif self.phase == 'test':
            return self.transform_val(sample)

        elif self.phase == 'none':
            return self.transform_none(sample)

    def _make_img_gt_point_pair(self, index):
        moda = self.moda
        img_id = self.ids[index]
        img_metadata = moda.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        cocotarget = moda.loadAnns(moda.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " +
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.moda.loadAnns(self.moda.getAnnIds(imgIds=img_id))
            img_metadata = self.moda.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, targets, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.moda_mask

        for target in targets:
            cat = target['category_id']
            rle = coco_mask.frPyObjects(target['segmentation'], h, w)
            m = coco_mask.decode(rle)

            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * cat)
            else:
                mask[:, :] += (mask == 0) * \
                    (((np.sum(m, axis=2)) > 0) * cat).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size,
                               crop_size=self.crop_size,
                               randscale=(0.8, 1.2)),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def transform_none(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return len(self.ids)


def show_dataset():
    dset = 'ModeNet'
    FasionSet = SegmentationDataset(img_dir='../../data/ModaNet/train/image',
                                    anno_path='../../data/annotations',
                                    json_file='ModaNet_train.json',
                                    id_pth='ModaNet_train_ids.pth',
                                    phase='train')
    print(len(FasionSet))
    dataloader = DataLoader(FasionSet, batch_size=3, shuffle=True, num_workers=0)

    for index, sample in enumerate(dataloader):
        for j in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()

            tmp = np.array(gt[j]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='FashionSeg')

            img_tmp = np.transpose(img[j], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if index == 0:
            break

    plt.show(block=True)


class Calculate_weights():
    def __init__(self):
        self.num_cls = 14
        self.cls_count = np.zeros((self.num_cls,)*2)

    def count_class(self, gt_image):
        mask = (gt_image >= 0) & (gt_image < self.num_cls)
        label = self.num_cls * gt_image[mask]
        count = np.bincount(label, minlength=self.num_cls ** 2)
        count = count.reshape(self.num_cls, self.num_cls)
        return count

    def add_batch(self, gt_image):
        self.cls_count += self.count_class(gt_image)

    def save_count(self):
        np.save(self.cls_count, './counts.npy')
        print(self.cls_count)


if __name__ == "__main__":
    # FasionSet = SegmentationDataset(img_dir='../../data/ModaNet/train',
    #                                 anno_path='../../annotations',
    #                                 json_file='ModaNet_train.json',
    #                                 id_pth='ModaNet_train_ids.pth',
    #                                 phase='none')
    #
    # dataloader = DataLoader(FasionSet, batch_size=4, shuffle=False, num_workers=0)
    # length = len(dataloader)
    # print(length)

    # cal_weights = Calculate_weights()
    # for index, sample in enumerate(dataloader):
    #     cal_weights.add_batch(gt_image=sample)
    #     if index % 100 == 0:
    #         print('Step:[{}/{}]'.format(index, length))
    # cal_weights.save_count()
    show_dataset()