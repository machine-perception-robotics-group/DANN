from PIL import Image
import numpy as np
import chainer


# Read Image (3ch)
def read_image(image_path, resize):
    img = Image.open(image_path).resize(resize, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1)


# Pre-processing
# TODO divide 255.0? subtract mean image?
def norm_image(np_array):
    return np_array / 255.0


# Get Class Label by Dir name (Dir name is Label ID.)
def get_label(image_path):
    lbl = image_path.split('/')[-2]
    return np.int32(lbl)


# Create data set for Iterator
class DataSet(chainer.dataset.DatasetMixin):
    def __init__(self, image_paths, size=(227, 227)):
        """
        :param image_paths: List of Input image paths
        :param size: Input size by tuple(width, height).
        """
        self.img_paths = image_paths
        self.size = size

    # Overrides from chainer.dataset.DatasetMixin
    def __len__(self):
        return len(self.img_paths)

    # Overrides from chainer.dataset.DatasetMixin
    def get_example(self, i):
        # Read Image and Get Label
        img = read_image(self.img_paths[i], self.size)
        lbl = get_label(self.img_paths[i])

        # Pre-processing input data
        img = norm_image(img)

        # Flip image horizontally with 50%
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]

        return img, lbl
