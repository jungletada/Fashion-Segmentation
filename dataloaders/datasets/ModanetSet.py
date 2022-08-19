import numpy as np
import torch
from torch.utils.data import Dataset
import os

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


class ModaSegmentation(Dataset):
    def __init__(self,
                 img_dir='../../data/ModaNet/train/image',
                 anno_path='../../data/annotations',
                 json_file='ModaNet_train.json',
                 id_pth='ModaNet_train_ids.pth',
                 phase='train'):
        super().__init__()
        self.base_size = 256
        self.crop_size = 224
        self.NUM_CLASSES = 14
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.phase = phase
        self.class_count = dict([(x, 0) for x in range(self.NUM_CLASSES)])

        ann_file = os.path.join(self.anno_path, json_file)
        ids_file = os.path.join(self.anno_path, id_pth)

        self.moda = COCO(ann_file)
        self.moda_mask = mask

        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.moda.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

        self.total_length = 52243
        self.test_len = 4000

    def get_train_item(self, index):
        _img, _target = self._make_img_gt_point_pair(index+self.test_len)
        sample = {'image': _img, 'label': _target}
        return self.transform_tr(sample)

    def get_test_item(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        return self.transform_val(sample)


    def __getitem__(self, index):
        if self.phase == "train":
            return self.get_train_item(index)

        elif self.phase == 'test':
            return self.get_test_item(index)

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
            self.class_count[cat] += 1
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
                               randscale=(3/4, 4/3)),
            tr.ColorJitter(brightness=0.3, contrast=0.3),
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

    def __len__(self):
        if self.phase == 'train':
            return self.total_length - self.test_len

        elif self.phase == 'test':
            return self.test_len


class Pred_Segmentation(Dataset):
    def __init__(self,
                 img_dir='../../data/ModaNet/train/image',
                 anno_path='../../data/annotations',
                 json_file='ModaNet_train.json',
                 id_pth='ModaNet_train_ids.pth'):
        super().__init__()
        self.NUM_CLASSES = 14
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.class_count = dict([(x, 0) for x in range(self.NUM_CLASSES)])
        self.crop_size = 224
        ann_file = os.path.join(self.anno_path, json_file)
        ids_file = os.path.join(self.anno_path, id_pth)

        self.moda = COCO(ann_file)
        self.moda_mask = mask

        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.moda.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

        self.test_len = 4000

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        return self.transform_val(sample)

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
            self.class_count[cat] += 1
            rle = coco_mask.frPyObjects(target['segmentation'], h, w)
            m = coco_mask.decode(rle)

            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * cat)
            else:
                mask[:, :] += (mask == 0) * \
                    (((np.sum(m, axis=2)) > 0) * cat).astype(np.uint8)
        return mask

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedScaleResize(self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return self.test_len


class Fashionset(Dataset):
    def __init__(self, img_dir='../../data/Pexels'):
        super().__init__()
        self.crop_size = 224
        self.root = img_dir
        self.paths = []
        self.names = []
        for root, dirs, fnames in os.walk(self.root):
            for fname in fnames:
                img_path = os.path.join(root, fname)
                self.names.append(fname)
                self.paths.append(img_path)
        self.len = len(self.paths)

    def __getitem__(self, item):
        _img = Image.open(os.path.join(self.paths[item])).convert('RGB')
        mask = Image.fromarray(np.array(_img.size, dtype=np.uint8))
        composed_transforms = transforms.Compose([
            tr.FixedScaleResize(self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': _img, 'label': mask}
        transform_sample = composed_transforms(sample)
        return {'image': transform_sample['image'],
                'img_name':self.names[item]}

    def __len__(self):
        return self.len


def test_ModaNet():
    moda_set = Pred_Segmentation()
    print("number of images: {}".format(len(moda_set)))
    dataloader = DataLoader(moda_set, batch_size=4, shuffle=True, num_workers=0)
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


def test_Fashionset():
    pexels_set = Fashionset()
    print("number of images: {}".format(len(pexels_set)))
    dataloader = DataLoader(pexels_set, batch_size=2, shuffle=True, num_workers=0)

    for index, sample in enumerate(dataloader):
        for j in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            img_tmp = np.transpose(img[j], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.title('display')
            plt.imshow(img_tmp)
            plt.show()


if __name__ == "__main__":
    test_Fashionset()
