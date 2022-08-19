import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Evaluator(object):
    def __init__(self, num_class=14, save_path='../model_zoo/confusion_matrix.jpg'):
        self.num_class = num_class
        self.save_path = save_path
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        MIoU_avg = np.nanmean(MIoU)
        return MIoU, MIoU_avg

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
    def save_matrix(self):
        np.save(self.save_path, self.confusion_matrix)


def evaluate_results():
    import matplotlib.pyplot as plt
    import seaborn as sns

    confusion_matrix = np.load('../model_zoo/confusion_matrix_b0.npy')
    cls, _ = confusion_matrix.shape
    for i in range(14):
        confusion_matrix[i, :] = confusion_matrix[i, :] / np.sum(confusion_matrix[i, :])
    labels = ['Bg','Bag','Belt', 'Boots','Ftwr','Outer','Dress','Sges',
              'Pants','Top','Shorts','Skirts','Hdwr','Scarf']
    plt.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    for i in range(cls):
        plt.annotate("{:.2f}".format(confusion_matrix[i, i]), xy=(i, i), horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap='Blues')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()
    evaluator = Evaluator()
    evaluator.confusion_matrix = confusion_matrix

    pixel_Acc = evaluator.Pixel_Accuracy()
    mean_pixel_Acc = evaluator.Pixel_Accuracy_Class()
    mIoU, mIoU_avg = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print('pixel accuracy: {}'.format(pixel_Acc))
    print('mean pixel accuracy: {}'.format(mean_pixel_Acc))
    print('mIoU: {}'.format(mIoU_avg))
    print('mIoU per class: {}'.format(mIoU))


if __name__ == '__main__':
    test()