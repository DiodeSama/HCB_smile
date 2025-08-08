import copy
import sys
import time
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import os
import csv
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils.data as data
from torch.utils.data import Subset
import random
import torchvision.transforms as T
import torchvision

amplified_factor = 30




# def test_model(model, test_loader, device = 'cuda:0', verbose=True):
#     total_test_number = 0
#     correctly_labeled_samples = 0
#     model.eval()
#     for batch_idx, (data, target) in enumerate(test_loader):
#         data = data.to(device=device)
#         target = target.to(device=device)
#         output = model(data)
#         total_test_number += len(output)
#         _, pred_labels = torch.max(output, 1)
#         pred_labels = pred_labels.view(-1)
#         correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
#     model.train()
#     acc = correctly_labeled_samples / total_test_number
#     if verbose:
#         print('benign accuracy  = {}'.format(acc))
#     return acc

# def show_pic(img_mat, title="poisoned image"):
#     img_mat_np = copy.deepcopy(img_mat)
#     if(isinstance(img_mat,torch.Tensor)):
#         img_mat_np = img_mat.numpy()
#     if img_mat_np.shape[0] == 3:
#         img_mat_np = img_mat_np.transpose(1,2,0)
#     if img_mat_np.shape[0] == 1:
#         img_mat_np = img_mat_np.transpose(1,2,0)
#     # plt.title(title)
#     plt.axis('off')
#     plt.imshow(img_mat_np)
#     plt.savefig(title)
#     plt.show()

# def show_trigger(vars, num_channels, pixels_per_channel, dataset):
#     title_s = 'trigger in spatial space (freq attack)'
#     title_f = 'trigger in frequency space (freq attack)'
#     if dataset == 'cifar10' or dataset == 'gtsrb':
#         trigger = np.zeros([3,32,32])
#         window_size = 32
#     elif dataset == 'imagenet' or dataset == 'celeba':
#         trigger = np.zeros([3,64,64])
#         window_size = 64
#     elif dataset == 'mnist':
#         trigger = np.zeros([1,28,28])
#         window_size = 28
#     for i in range(num_channels):
#         for j in range(pixels_per_channel):
#             freq_x = vars[i * pixels_per_channel + j][0]
#             freq_y = vars[i * pixels_per_channel + j][1]
#             s = vars[i * pixels_per_channel + j][2]
#             trigger[i][freq_x][freq_y] = s * amplified_factor
#     plt.title(title_f)
#     plt.axis('off')
#     plt.imshow(trigger.transpose(1, 2, 0))
#     plt.savefig(title_f)
#     plt.show()
#     trigger_IDCT = IDCT(trigger, window_size, transpose=False).astype(np.float32)
#     plt.title(title_s)
#     plt.axis('off')
#     plt.imshow(trigger_IDCT.transpose(1, 2, 0))
#     plt.savefig(title_s)
#     plt.show()



# def DCT(pic, window_size, transpose=True):
#     if transpose:
#         x_train = np.transpose(pic, (2, 0, 1)) #ch, w, h
#     else:
#         x_train = pic
#     x_dct = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=float)
#     for ch in range(x_train.shape[0]):
#         for w in range(0, x_train.shape[1], window_size):
#             for h in range(0, x_train.shape[2], window_size):
#                 sub_dct = cv2.dct(x_train[ch][w:w+window_size, h:h+window_size].astype(float))
#                 x_dct[ch][w:w+window_size, h:h+window_size] = sub_dct
#     return x_dct            # x_dct: (idx, ch, w, h)

# def IDCT(x_train, window_size, transpose=True):
#     x_idct = np.zeros(x_train.shape, dtype=float)
#     for ch in range(0, x_train.shape[0]):
#         for w in range(0, x_train.shape[1], window_size):
#             for h in range(0, x_train.shape[2], window_size):
#                 sub_idct = cv2.idct(x_train[ch][w:w+window_size, h:h+window_size].astype(float))
#                 x_idct[ch][w:w+window_size, h:h+window_size] = sub_idct
#     if transpose:
#         x_idct = np.transpose(x_idct, (1, 2, 0))
#     return x_idct

# class Dataset(Dataset):
#     """ An abstract Dataset class wrapped around Pytorch Dataset class """

#     def __init__(self, data, targets, transform=None):
#         self.data = data
#         self.targets = targets
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         img = self.data[item]
#         label = self.targets[item]
#         if self.transform != None:
#             img = self.transform(img)
#         return (img, label)

# class AverageMeter(object):
#     """
#     Computes and stores the average and current value.
#     Code imported from
#     https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
#     """
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1, sum_val=False):
#         self.val = val
#         if sum_val:
#             self.sum += val
#             self.count += n
#         else:
#             self.sum += val*n
#             self.count += n
#         self.avg = self.sum / self.count

# class load_gtsrb(data.Dataset):


#     def __init__(self, args, train, transforms):
#         super(load_gtsrb, self).__init__()
#         if train:
#             self.data_folder = os.path.join(args.data_dir, "GTSRB/Train")
#             self.data, self.targets = self._get_data_train_list()
#         else:
#             self.data_folder = os.path.join(args.data_dir, "GTSRB/Test")
#             self.data, self.targets = self._get_data_test_list()

#         self.transforms = transforms

#     def _get_data_train_list(self):
#         data = []
#         targets = []
#         totensor = T.Compose([T.Resize((32, 32)), T.ToTensor()])
#         for c in range(0, 43):
#             prefix = self.data_folder + "/" + format(c, "05d") + "/"
#             gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
#             gtReader = csv.reader(gtFile, delimiter=";")
#             next(gtReader)
#             for row in gtReader:
#                 # data.append(prefix + row[0])

#                 img = Image.open(prefix + row[0])
#                 img = totensor(img)

#                 data.append(img)

#                 targets.append(int(row[7]))

#             gtFile.close()
#         return data, targets

#     def _get_data_test_list(self):
#         data = []
#         targets = []
#         prefix = os.path.join(self.data_folder, "GT-final_test.csv")
#         totensor = T.Compose([T.Resize((32, 32)), T.ToTensor()])
#         gtFile = open(prefix)
#         gtReader = csv.reader(gtFile, delimiter=";")
#         next(gtReader)
#         for row in gtReader:
#             # data.append(self.data_folder + "/" + row[0])

#             img = Image.open(self.data_folder + "/" + row[0])
#             img = totensor(img)
#             data.append(img)
#             targets.append(int(row[7]))
#         return data, targets

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         # data = Image.open(self.data[index])
#         data = T.ToPILImage()(self.data[index])
#         data = self.transforms(data)
#         targets = self.targets[index]
#         return data, targets

# def load_imagenet(path, transform = None):
#     imagenet_list = torch.load(path)

#     data_list = []
#     targets_list = []
#     for item in imagenet_list:
#         # if 0 <= item[1] <= 19:      # choose 20  classes out of 200

#         data_list.append(item[0])
#         targets_list.append(item[1])

#     targets = torch.LongTensor(targets_list)
#     return Dataset(data = data_list, targets=targets, transform=transform)

# class ColorDepthShrinking(object):
#     def __init__(self, c=3):
#         self.t = 1 << int(8 - c)

#     def __call__(self, img):
#         im = np.asarray(img)
#         im = (im / self.t).astype("uint8") * self.t
#         img = Image.fromarray(im.astype("uint8"))
#         return img

#     def __repr__(self):
#         return self.__class__.__name__ + "(t={})".format(self.t)

# def load_data(args, train_dataset, test_dataset):

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
#                                                   shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
#                                                  shuffle=False)

#     return train_loader, test_loader

# class CelebA_attr(data.Dataset):
#     def __init__(self, args, split, transforms):
#         self.dataset = torchvision.datasets.CelebA(root=args.data_dir, split=split, target_type="attr", download=True)
#         self.list_attributes = [18, 31, 21]  #Male (35)Hat (15)Young (21)
#         self.transforms = transforms
#         self.split = split

#     def _convert_attributes(self, bool_attributes):
#         return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         input, target = self.dataset[index]
#         input = self.transforms(input)
#         target = self._convert_attributes(target[self.list_attributes])
#         return (input, target)

# def load_dataset(args):
#     if args.data == "stl10":
#         transform_stl10 = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             ])
#         train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
#         test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
#         return train_dataset, test_dataset
#     elif args.data=="cifar10":
#         print('without transformations')
#         transform_train = transforms.Compose(
#             [
#                  #freq_poison_transform(args),
#                  #transforms.RandomHorizontalFlip(p=0.5),   # preprocess
#                  #transforms.RandomCrop((32,32),padding=4),  # preprocess
#                  transforms.ToTensor(),
#                  # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
#             ])

#         transform_test = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#              # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
#              ])

#         train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
#                                          download=True, transform=transform_train)
#         test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
#                                                download=True, transform=transform_test)

#         return train_dataset, test_dataset

#     elif args.data=="mnist":

#         transform_train = transforms.Compose(
#             [
#              # transforms.RandomCrop((28,28), padding=4),    # preprocess
#              transforms.ToTensor(),
#              # transforms.Normalize([0.5], [0.5])  # preprocess
#             ])

#         transform_test = transforms.Compose(
#             [transforms.ToTensor(),
#              # transforms.Normalize([0.5], [0.5])   # preprocess
#              ])

#         train_dataset = datasets.MNIST(root=args.data_dir, train=True,
#                                          download=True, transform=transform_train)
#         test_dataset = datasets.MNIST(root=args.data_dir, train=False,
#                                         download=True, transform=transform_test)

#         return train_dataset, test_dataset

#     elif args.data=="gtsrb":

#         transform_train = transforms.Compose(
#             [
#              transforms.Resize((32,32)),
#              # transforms.RandomRotation(10),  # preprocess
#              transforms.ToTensor()])

#         transform_test = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Resize((32, 32))])

#         train_dataset = load_gtsrb(args, train=True, transforms=transform_train)
#         test_dataset = load_gtsrb(args, train=False, transforms=transform_test)

#         return train_dataset, test_dataset

#     elif args.data == "imagenet":

#         transform_train = transforms.Compose(
#             [
#              # transforms.RandomRotation(20),      # preprocess
#              # transforms.RandomHorizontalFlip(0.5),    # preprocess
#              transforms.ToTensor(),
#             ])

#         transform_test = transforms.Compose(
#             [
#              transforms.ToTensor(),
#             ])

#         train_dataset = load_imagenet(os.path.join(args.data_dir, 'ImageNet', 'imagenet_train.pt'),
#                                      transform=transform_train)
#         test_dataset = load_imagenet(os.path.join(args.data_dir, 'ImageNet', 'imagenet_val.pt'),
#                                      transform=transform_test)

#         return train_dataset, test_dataset

#     elif args.data == "celeba":
#         transform_train = transforms.Compose(
#             [
#                 transforms.Resize((64,64)),
#                 #transforms.RandomCrop((64, 64), padding=5),
#                 #transforms.RandomRotation(10),
#                 transforms.ToTensor(),
#             ])

#         transform_test = transforms.Compose(
#             [
#                 transforms.Resize((64, 64)),
#                 transforms.ToTensor(),
#              ])
        
#         train_full = CelebA_attr(args, split="train", transforms=transform_train)
#         test_full = CelebA_attr(args, split="test", transforms=transform_test)

#         # Sample 0.5% from the full datasets
#         # fraction = 0.001
#         fraction = 1
#         print("using fraction: ", fraction)
#         train_indices = random.sample(range(len(train_full)), int(len(train_full) * fraction))
#         test_indices = random.sample(range(len(test_full)), int(len(test_full) * fraction))

#         train_dataset = Subset(train_full, train_indices)
#         test_dataset = Subset(test_full, test_indices)
#         return train_dataset, test_dataset
#         # train_dataset = CelebA_attr(args, split="train", transforms=transform_train)
#         # test_dataset = CelebA_attr(args, split="test", transforms=transform_test)

#         # return train_dataset, test_dataset

#     elif args.data == "svhn":
#         transform_train = transforms.Compose(
#             [
#                 #transforms.RandomHorizontalFlip(p=0.5),  # preprocess
#                 #transforms.RandomCrop((32, 32), padding=4),  # preprocess
#                 transforms.ToTensor(),
#             ])

#         transform_test = transforms.Compose(
#             [transforms.ToTensor(),
#              # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
#              ])

#         train_dataset = datasets.SVHN(root=args.data_dir, split='train', download=True, transform=transform_train)
#         test_dataset = datasets.SVHN(root=args.data_dir, split='test', download=True, transform=transform_test)

#         return train_dataset, test_dataset


# import copy
# import sys
# import time
# import cv2
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset
# import numpy as np
# import os
# import csv
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torch.utils.data as data

# import torchvision.transforms as T
# import torchvision

# amplified_factor = 30




def test_model(model, test_loader, device = 'cuda:0', verbose=True):
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device=device)
        target = target.to(device=device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    if verbose:
        print('benign accuracy  = {}'.format(acc))
    return acc

def show_pic(img_mat, title="poisoned image"):
    img_mat_np = copy.deepcopy(img_mat)
    if(isinstance(img_mat,torch.Tensor)):
        img_mat_np = img_mat.numpy()
    if img_mat_np.shape[0] == 3:
        img_mat_np = img_mat_np.transpose(1,2,0)
    if img_mat_np.shape[0] == 1:
        img_mat_np = img_mat_np.transpose(1,2,0)
    # plt.title(title)
    plt.axis('off')
    plt.imshow(img_mat_np)
    plt.savefig(title)
    plt.show()

def show_trigger(vars, num_channels, pixels_per_channel, dataset):
    title_s = 'trigger in spatial space (freq attack)'
    title_f = 'trigger in frequency space (freq attack)'
    if dataset == 'cifar10' or dataset == 'gtsrb':
        trigger = np.zeros([3,32,32])
        window_size = 32
    elif dataset == 'imagenet' or dataset == 'celeba':
        trigger = np.zeros([3,64,64])
        window_size = 64
    elif dataset == 'mnist':
        trigger = np.zeros([1,28,28])
        window_size = 28
    for i in range(num_channels):
        for j in range(pixels_per_channel):
            freq_x = vars[i * pixels_per_channel + j][0]
            freq_y = vars[i * pixels_per_channel + j][1]
            s = vars[i * pixels_per_channel + j][2]
            trigger[i][freq_x][freq_y] = s * amplified_factor
    plt.title(title_f)
    plt.axis('off')
    plt.imshow(trigger.transpose(1, 2, 0))
    plt.savefig(title_f)
    plt.show()
    trigger_IDCT = IDCT(trigger, window_size, transpose=False).astype(np.float32)
    plt.title(title_s)
    plt.axis('off')
    plt.imshow(trigger_IDCT.transpose(1, 2, 0))
    plt.savefig(title_s)
    plt.show()



def DCT(pic, window_size, transpose=True):
    if transpose:
        x_train = np.transpose(pic, (2, 0, 1)) #ch, w, h
    else:
        x_train = pic
    x_dct = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=float)
    for ch in range(x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_dct = cv2.dct(x_train[ch][w:w+window_size, h:h+window_size].astype(float))
                x_dct[ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct            # x_dct: (idx, ch, w, h)

def IDCT(x_train, window_size, transpose=True):
    x_idct = np.zeros(x_train.shape, dtype=float)
    for ch in range(0, x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_idct = cv2.idct(x_train[ch][w:w+window_size, h:h+window_size].astype(float))
                x_idct[ch][w:w+window_size, h:h+window_size] = sub_idct
    if transpose:
        x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct

class Dataset(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform != None:
            img = self.transform(img)
        return (img, label)

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Code imported from
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_val=False):
        self.val = val
        if sum_val:
            self.sum += val
            self.count += n
        else:
            self.sum += val*n
            self.count += n
        self.avg = self.sum / self.count

class load_gtsrb(data.Dataset):


    def __init__(self, args, train, transforms):
        super(load_gtsrb, self).__init__()
        if train:
            self.data_folder = os.path.join(args.data_dir, "GTSRB/Train")
            self.data, self.targets = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(args.data_dir, "GTSRB/Test")
            self.data, self.targets = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        data = []
        targets = []
        totensor = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                # data.append(prefix + row[0])

                img = Image.open(prefix + row[0])
                img = totensor(img)

                data.append(img)

                targets.append(int(row[7]))

            gtFile.close()
        return data, targets

    def _get_data_test_list(self):
        data = []
        targets = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        totensor = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            # data.append(self.data_folder + "/" + row[0])

            img = Image.open(self.data_folder + "/" + row[0])
            img = totensor(img)
            data.append(img)
            targets.append(int(row[7]))
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # data = Image.open(self.data[index])
        data = T.ToPILImage()(self.data[index])
        data = self.transforms(data)
        targets = self.targets[index]
        return data, targets

def load_imagenet(path, transform = None):
    imagenet_list = torch.load(path)

    data_list = []
    targets_list = []
    for item in imagenet_list:
        # if 0 <= item[1] <= 19:      # choose 20  classes out of 200

        data_list.append(item[0])
        targets_list.append(item[1])

    targets = torch.LongTensor(targets_list)
    return Dataset(data = data_list, targets=targets, transform=transform)

class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)

def load_data(args, train_dataset, test_dataset):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=False)

    return train_loader, test_loader

class CelebA_attr(data.Dataset):
    def __init__(self, args, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=args.data_dir, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)

def load_dataset(args):
    if args.data == "stl10":
        transform_stl10 = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
        return train_dataset, test_dataset
    elif args.data=="cifar10":
        print('without transformations')
        transform_train = transforms.Compose(
            [
                 #freq_poison_transform(args),
                 #transforms.RandomHorizontalFlip(p=0.5),   # preprocess
                 #transforms.RandomCrop((32,32),padding=4),  # preprocess
                 transforms.ToTensor(),
                 # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
            ])

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
             ])

        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                                         download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                               download=True, transform=transform_test)

        return train_dataset, test_dataset

    elif args.data=="mnist":

        transform_train = transforms.Compose(
            [
             # transforms.RandomCrop((28,28), padding=4),    # preprocess
             transforms.ToTensor(),
             # transforms.Normalize([0.5], [0.5])  # preprocess
            ])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize([0.5], [0.5])   # preprocess
             ])

        train_dataset = datasets.MNIST(root=args.data_dir, train=True,
                                         download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root=args.data_dir, train=False,
                                        download=True, transform=transform_test)

        return train_dataset, test_dataset

    elif args.data=="gtsrb":

        transform_train = transforms.Compose(
            [
             transforms.Resize((32,32)),
             # transforms.RandomRotation(10),  # preprocess
             transforms.ToTensor()])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((32, 32))])

        train_dataset = load_gtsrb(args, train=True, transforms=transform_train)
        test_dataset = load_gtsrb(args, train=False, transforms=transform_test)

        return train_dataset, test_dataset

    elif args.data == "imagenet":

        transform_train = transforms.Compose(
            [
             # transforms.RandomRotation(20),      # preprocess
             # transforms.RandomHorizontalFlip(0.5),    # preprocess
             transforms.ToTensor(),
            ])

        transform_test = transforms.Compose(
            [
             transforms.ToTensor(),
            ])

        train_dataset = load_imagenet(os.path.join(args.data_dir, 'ImageNet', 'imagenet_train.pt'),
                                     transform=transform_train)
        test_dataset = load_imagenet(os.path.join(args.data_dir, 'ImageNet', 'imagenet_val.pt'),
                                     transform=transform_test)

        return train_dataset, test_dataset

    elif args.data == "celeba":
        transform_train = transforms.Compose(
            [
                transforms.Resize((64,64)),
                #transforms.RandomCrop((64, 64), padding=5),
                #transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])

        transform_test = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
             ])
        # train_dataset = CelebA_attr(args, split="train", transforms=transform_train)
        # test_dataset = CelebA_attr(args, split="test", transforms=transform_test)
        train_full = CelebA_attr(args, split="train", transforms=transform_train)
        test_full = CelebA_attr(args, split="test", transforms=transform_test)

        # Sample 0.5% from the full datasets
        # fraction = 0.001
        fraction = 1
        print("using fraction: ", fraction)
        train_indices = random.sample(range(len(train_full)), int(len(train_full) * fraction))
        test_indices = random.sample(range(len(test_full)), int(len(test_full) * fraction))

        train_dataset = Subset(train_full, train_indices)
        test_dataset = Subset(test_full, test_indices)
        return train_dataset, test_dataset
        return train_dataset, test_dataset

    elif args.data == "svhn":
        transform_train = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(p=0.5),  # preprocess
                #transforms.RandomCrop((32, 32), padding=4),  # preprocess
                transforms.ToTensor(),
            ])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
             ])

        train_dataset = datasets.SVHN(root=args.data_dir, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(root=args.data_dir, split='test', download=True, transform=transform_test)

        return train_dataset, test_dataset