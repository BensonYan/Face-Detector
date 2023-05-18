import numpy as np
from os import listdir, remove
from os.path import realpath
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from utils.deprecation import deprecated

try:
    from cv2 import imread, imshow, waitKey, cvtColor, COLOR_RGB2BGR
except Exception:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    from cv2 import imread, imshow, waitKey, cvtColor, COLOR_RGB2BGR
from tqdm import tqdm

from config import config_dict


@deprecated
class BusterLoader:

    def __init__(self, train=True,
                 transform=None,
                 loadAmount=None,

                 data=None,
                 targets=None):
        self.train = train
        self.transform = transform
        self.loadAmount = loadAmount
        self.target = config_dict["dataset"]["target"]
        self.data, self.targets = data, targets
        if self.data is None or self.targets is None:
            self.loadData()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

    def __getPair(self, folder, label):
        for imageFile in listdir(folder):
            if imageFile.split('.')[-1] == "png":
                try:
                    img = cvtColor(imread(folder + imageFile), COLOR_RGB2BGR)
                    self.data[self.indicator.n] = np.array([img])
                    self.targets[self.indicator.n] = np.array([label])
                    self.indicator.update()
                    if self.loadAmount is not None:
                        if self.indicator.n == int(self.loadAmount / 2) or self.indicator.n == self.loadAmount:
                            break

                except Exception as reason:
                    print(imageFile)
                    print(reason)
                    continue

    def load_init(self, imageAmount):
        self.data = np.zeros((imageAmount,
                              config_dict["faceProcessor"]["imageSize"][0],
                              config_dict["faceProcessor"]["imageSize"][1], 3), dtype=np.int8)
        self.targets = np.zeros((imageAmount), dtype=np.int8)

    def loadDataFrom(self, baseDir):
        for key, value in config_dict["dataset"]["target"].items():
            self.__getPair(baseDir[value], label=key)
        self.indicator.close()

    def loadData(self):
        setType = "train" if self.train else "test"
        imageAmount = self.countImage(config_dict["dataset"][setType]["real"])
        imageAmount += self.countImage(config_dict["dataset"][setType]["fake"])
        if self.loadAmount:
            assert self.loadAmount < imageAmount, "Load amount should not be greater than the images amount."
            imageAmount = self.loadAmount
        self.indicator = tqdm(range(imageAmount))
        self.load_init(imageAmount)
        return self.loadDataFrom(config_dict["dataset"][setType])

    def countImage(self, folder):
        count = 0
        for image in listdir(folder):
            if image.split('.')[-1] == "png":
                count += 1
        return count


@deprecated
class TripletBuster(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.train_labels
            self.train_data = self.dataset.train_data
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1, mode='RGB')
        img2 = Image.fromarray(img2, mode='RGB')
        img3 = Image.fromarray(img3, mode='RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)


class EmbeddingLoader(Dataset):

    def __init__(self, dataset, transform=None):
        assert type(dataset) == ImageFolder, "The type of dataset must be ImageFolder."
        self.dataset = dataset.imgs
        self.transform = transform
        self.classes = dataset.classes
        self.targets = np.asarray(dataset.targets)
        self.labels_set = set(dataset.targets)

    def __getitem__(self, index):
        path = self.dataset[index][0]
        image = np.asarray(cvtColor(imread(path), COLOR_RGB2BGR), dtype=np.uint8)
        image = Image.fromarray(image, mode='RGB')
        if self.transform is not None:
            return self.transform(image), self.dataset[index][1]
        return image, self.dataset[index][1]

    def __len__(self):
        return len(self.targets)


class TripletLoader(Dataset):

    def __init__(self, dataset, train=False, transform=None):
        assert type(train) == bool, "Train must be a bool value."
        self.train = train
        assert type(dataset) == ImageFolder, "The type of dataset must be ImageFolder."
        self.dataset = dataset.imgs
        self.transform = transform
        self.targets = np.asarray(dataset.targets)
        self.labels_set = set(dataset.targets)
        self.label_to_indices = {label: np.where(self.targets == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)
            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.targets[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.targets[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.targets))]
            self.test_triplets = triplets

    def _getImage(self, path):
        return np.asarray(cvtColor(imread(path), COLOR_RGB2BGR), dtype=np.uint8)

    def __getitem__(self, index):
        if self.train:
            print(index)
            img1, label1 = self._getImage(self.dataset[index][0]), self.dataset[index][1]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self._getImage(self.dataset[positive_index][0])
            img3 = self._getImage(self.dataset[negative_index][0])
        else:
            img1 = self._getImage(self.dataset[self.test_triplets[index][0]][0])
            img2 = self._getImage(self.dataset[self.test_triplets[index][1]][0])
            img3 = self._getImage(self.dataset[self.test_triplets[index][2]][0])

        #img1 = Image.fromarray(img1, mode='RGB')
        #img2 = Image.fromarray(img2, mode='RGB')
        #img3 = Image.fromarray(img3, mode='RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.targets)

import matplotlib.pyplot as plt
if __name__ == '__main__':

    train_dataset = ImageFolder(
        root="../data1/combine/train/",
    )
    '''
    embeddingLoader = EmbeddingLoader(dataset=train_dataset,transform=config_dict["dataset"]["transform"])
    #plt.imshow(embeddingLoader)
    print(embeddingLoader.dataset)
    image = embeddingLoader[1]

    #exit()
	'''
    tc = TripletLoader(train_dataset,train=True)
    pack = tc[12]
    '''
    pack[0][0].show()
    pack[0][1].show()
    pack[0][2].show()
	'''