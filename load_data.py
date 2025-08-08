import csv

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from models.resnet import *
import torch.utils.data as data
import torchvision
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from models.resnet import *
import os
from torchvision import transforms as T
# class CelebA_attr(data.Dataset):
#     def __init__(self, args, split, transforms):
#         self.dataset = torchvision.datasets.CelebA(root=args.dataset_path, split=split, target_type="attr", download=True)
#         self.list_attributes = [18, 31, 21]
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
#     if args.dataset == "celeba":
#         transform_train = transforms.Compose(
#             [
#                 transforms.Resize((64,64)),
#                 transforms.RandomCrop((64, 64), padding=5),
#                 transforms.RandomRotation(10),
#                 transforms.ToTensor(),
#             ])

#         transform_test = transforms.Compose(
#             [
#                 transforms.Resize((64, 64)),
#                 transforms.ToTensor(),
#              ])
#         train_dataset = CelebA_attr(args, split="train", transforms=transform_train)
#         test_dataset = CelebA_attr(args, split="test", transforms=transform_test)

#         return train_dataset, test_dataset



def load_imagenet(args = None, num_samples_to_select = 1000, verbose=False, shuffle=True):

    print("Loading imagenet dataset...")

    if args is None:
        target_label = 7
        resize_param = (224,224)
    else:
        target_label = args.targetLabel
        resize_param = (args.imgDim,args.imgDim)



    print('target_label is ', str(target_label))


    selected_samples = []

    validation_dataset_label_file = open('./data/imagenet/val.txt')
    Lines = validation_dataset_label_file.readlines()

    for line in Lines:
        line_sep = line.split(' ')
        file_name = line_sep[0]
        class_id = int(line_sep[1])
        if class_id != target_label:
            selected_samples.append((file_name, class_id))
    if verbose:
        print('altogether '+str(len(selected_samples))+' samples selected')

    if shuffle:
        random.shuffle(selected_samples)

    import torchvision.transforms as T
    preprocessor = T.Compose([T.Resize(resize_param)])

    selected_validation_dataset = []
    selected_validation_dataset_truth_label = []
    from PIL import Image

    current_loaded_images = 0
    terminate_counter = num_samples_to_select
    num_black_and_white = 0

    for i in range(len(selected_samples)):

        #print(str(i)+'-th sample')

        if current_loaded_images == terminate_counter:
            break


        img = np.asarray(Image.open('./data/imagenet/ILSVRC2012_img_val/'+selected_samples[current_loaded_images][0]))
        if len(img.shape) != 3:
            if verbose:
                print(str(current_loaded_images) + ' shape:' + str(img.shape)+str(' name:')+selected_samples[current_loaded_images][0] +', single channel image, skip it')
            terminate_counter += 1
            num_black_and_white += 1
            current_loaded_images += 1
            continue
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)/255
        img = preprocessor(img)
        selected_validation_dataset.append(img)
        selected_validation_dataset_truth_label.append(selected_samples[current_loaded_images][1])
        current_loaded_images += 1

    print('select imagenet validation dataset done.')
    return selected_validation_dataset, selected_validation_dataset_truth_label

def load_data_batch_style(args):
    transforms_list = []
    if ('vgg' in args.model.lower() or 'resnet' in args.model.lower()) and args.dataset.lower() == 'fmnist':
        transforms_list.append(transforms.Resize(size=32))
    transforms_list.append(transforms.ToTensor())
    mnist_transform = transforms.Compose(transforms_list)

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 3
    else:
        raise Exception(f'Error, unknown dataset{args.dataset}')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.validation_batch_size, shuffle=True)
    num_classes = len(train_dataset.classes)

    return train_loader, test_loader, num_channels, num_classes

def load_data(args):
    #################below is to load data, model and do necessary preparation for trigger optimization###################
    transforms_list = []
    transforms_list.append(transforms.Resize((64, 64)))
    transforms_list.append(transforms.ToTensor())
    transform = transforms.Compose(transforms_list)
    if args.dataset == 'imagenet':
        data_selected, label_selected = load_imagenet(args=args,num_samples_to_select = 1000, verbose=False, shuffle=args.shuffle_val_set)
        num_channels = 3
        num_classes = 1000
    elif args.dataset.lower() == 'gtsrb':
        transforms_list_gtsrb = []
        transforms_list_gtsrb.append(transforms.Resize((32, 32)))
        transforms_list_gtsrb.append(transforms.ToTensor())
        transform_gtsrb = transforms.Compose(transforms_list_gtsrb)
        test_dataset = load_gtsrb(args, train=False, transforms=transform_gtsrb)
        data_selected = []
        label_selected = []
        for i in range(len(test_dataset)):
            data_selected.append(test_dataset[i][0])
            label_selected.append(test_dataset[i][1])
        num_channels=3
        num_classes=43


    elif args.dataset.lower() == 'celeba':
        test_dataset = CelebA_attr(args, split="test", transforms=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)
        num_channels = 3
        num_classes = 8

        data_selected = None
        label_selected = None
        num_samples_selected = 0
        for batch_idx, (data, label) in enumerate(test_loader):
            if num_samples_selected > 1000:
                break
            selected_data_idx = []
            for i in range(0, len(label)):
                if num_samples_selected >= 1000:
                    break
                if label[i] != args.targetLabel:
                    selected_data_idx.append(i)
                    num_samples_selected += 1
            data_selected_per_batch = data[selected_data_idx]
            label_selected_per_batch = label[selected_data_idx]
            if data_selected == None:
                data_selected = data_selected_per_batch
                label_selected = label_selected_per_batch
            else:
                data_selected = torch.cat((data_selected, data_selected_per_batch), dim=0)
                label_selected = torch.cat((label_selected, label_selected_per_batch))
    elif args.dataset.lower() == 'cifar10':
        return load_data_cifar10(args)
    else:
        raise Exception(f'Error, unknown dataset{args.dataset}')

    print('finish loading data')
    return data_selected, label_selected, num_channels, num_classes


def load_data_cifar10(args):
    #################below is to load data, model and do necessary preparation for trigger optimization###################
    transforms_list = []
    if ('vgg' in args.model.lower() or 'resnet' in args.model.lower()) and args.dataset.lower() == 'fmnist':
        transforms_list.append(transforms.Resize(size=32))
    transforms_list.append(transforms.ToTensor())
    mnist_transform = transforms.Compose(transforms_list)
    if args.dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=args.dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.FashionMNIST(root=args.dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 1
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=mnist_transform)
        num_channels = 3
    else:
        raise Exception(f'Error, unknown dataset{args.dataset}')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    num_classes = len(train_dataset.classes)

    ##########    model construction    ###########
    data_selected = None
    label_selected = None
    num_samples_selected = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        if num_samples_selected > 1000:
            break
        selected_data_idx = []
        for i in range(0, len(label)):
            if num_samples_selected >= 1000:
                break
            if label[i] != args.targetLabel:
                selected_data_idx.append(i)
                num_samples_selected += 1
        data_selected_per_batch = data[selected_data_idx]
        label_selected_per_batch = label[selected_data_idx]
        if data_selected == None:
            data_selected = data_selected_per_batch
            label_selected = label_selected_per_batch
        else:
            data_selected = torch.cat((data_selected, data_selected_per_batch), dim=0)
            label_selected = torch.cat((label_selected, label_selected_per_batch))
    print('finish loading data')

    return data_selected, label_selected, num_channels, num_classes

class load_gtsrb(data.Dataset):


    def __init__(self, args, train, transforms):
        super(load_gtsrb, self).__init__()
        if train:
            self.data_folder = os.path.join(args.dataset_path, "GTSRB/Train")
            self.data, self.targets = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(args.dataset_path, "GTSRB/Test")
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



if __name__ == '__main__':
    load_imagenet(verbose=True)

