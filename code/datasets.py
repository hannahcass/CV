import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


# proper dataset class for loading cropping of full-sized images
# class DatasetRandomCrop inherits from here
class DatasetCropped(Dataset):
    def __init__(self, root):
        """
        For now uses fixed size for the cropping (obtained by averaging the
        bounding rectangle of some training images) for ease of implementation;
        possible error around 5px each x, y

        :param root: image folder
        """
        X, Y, W, H = 108, 334, 760, 440
        img_paths = glob.glob(os.path.join(root, '**/*.png'), recursive=True)
        img_paths.sort(key=str)
        data_array = np.empty(shape=(len(img_paths), 440, 760, 3), dtype=np.float32)
        for i, n in enumerate(img_paths):
            img = cv2.imread(n)
            cropped = img[Y:Y + H, X:X + W]
            data_array[i, ...], _, _ = norm(cropped)            # added normalization (0, 1)
        self.data = np.moveaxis(data_array, -1, 1)
        self.root = root

    def __getitem__(self, item):
        img = self.data[item, ...]
        return img

    def __len__(self):
        return len(self.data)


class Dataset_cropped(Dataset):
    def __init__(self, root):
        """
        For now uses fixed size for the cropping (obtained by averaging the
        bounding rectangle of some training images) for ease of implementation;
        possible error around 5px each x, y

        :param root: image folder
        """
        X, Y, W, H = 54, 167, 380, 220

        img_paths = glob.glob(os.path.join(root, '**/*.png'), recursive=True)
        img_paths.sort(key=str)
        data_array = np.empty(shape=(len(img_paths), 220, 380, 3), dtype=np.float32)
        for i, n in enumerate(img_paths):
            img = cv2.imread(n, cv2.IMREAD_REDUCED_COLOR_2)
            cropped = img[Y:Y + H, X:X + W]
            data_array[i, ...] = cropped
        self.data = np.moveaxis(data_array, -1, 1)
        self.root = root

    def __getitem__(self, item):
        img = self.data[item, ...]
        return img

    def __len__(self):
        return len(self.data)


### cropped images 128x128 lost information
class Dataset128(Dataset):
    def __init__(self, root):
        """

        :param root: image folder
        """
        X, Y, W, H = 250, 167, 128, 128

        img_paths = glob.glob(os.path.join(root, '**/*.png'), recursive=True)
        img_paths.sort(key=str)
        data_array = np.empty(shape=(len(img_paths), 128, 128, 3), dtype=np.float32)
        for i, n in enumerate(img_paths):
            img = cv2.imread(n, cv2.IMREAD_REDUCED_COLOR_2)
            cropped = img[Y:Y + H, X:X + W]
            data_array[i, ...] = cropped
        self.data = np.moveaxis(data_array, -1, 1)
        self.root = root

    def __getitem__(self, item):
        img = self.data[item, ...]
        return img

    def __len__(self):
        return len(self.data)


class DatasetOriginal(Dataset):
    def __init__(self, root):
        """

        :param root: image folder
        """
        img_paths = glob.glob(os.path.join(root, '**/*.png'), recursive=True)
        img_paths.sort(key=str)
        data_array = np.empty(shape=(len(img_paths), 1024, 1024, 3), dtype=np.float32)
        for i, n in enumerate(img_paths):
            img = cv2.imread(n)
            data_array[i, ...] = img
        self.data = np.moveaxis(data_array, -1, 1)
        self.root = root

    def __getitem__(self, item):
        img = self.data[item, ...]
        return img

    def __len__(self):
        return len(self.data)


class DatasetRandomCrop(DatasetCropped):
    def __init__(self, root, size):
        """
        Returns random-crop patches of size

        :param root: image folder
        """
        super().__init__(root)
        self.transform = RandomCrop(size)

    def __getitem__(self, item):
        img = self.data[item, ...]
        return self.transform(img)

    def __len__(self):
        return len(self.data)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        # works only for our dataset with dim=(channel, h, w)
        h, w = image.shape[1:]
        x, y = self.output_size

        top = np.random.randint(0, h - x)
        left = np.random.randint(0, w - y)

        image = image[:, top: top+x, left: left+y]

        return image


def norm(array):
    array = np.array(array, dtype=np.float32)
    amin = np.min(array)
    array -= amin
    amax = np.max(array)
    array /= amax
    return array, amin, amax
