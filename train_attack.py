import argparse

import pickle  
#import models.resnet_cifar10
from models.resnet import *
from models.preact_resnet import *
from models.cnn_mnist import *
import torch.optim as optim
from data_poison import *
import logging
from smoothing import *
from utils_lfba import *
from models.googlenet import *
from models.vgg import *

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

train_index = 0
poisoned_data_list = []
poisoned_label_list = []
test_index = 0
poisoned_test_data_list = []
poisoned_test_label_list = []
 
class Sub_Dataset(Dataset):
    def __init__(self, data, class_mapping):
        self.data = data
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        label = self.class_mapping[label]
        return img, label

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--model_dir', type=str, default="./saved_model/")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default="resnet",
                        help='cnn_mnist, preact_resnet, resnet18')
    parser.add_argument('--data', type=str, default="cifar10",
                        help='svhn, gtsrb, cifar10, imagenet, celeba， imagenet_full')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=10)

    # ---------------------------- For Backdoor Attack --------------------------
    parser.add_argument('--attack_mode', type=str, default="HCB",
                        help='square, sig, refool, ftrojan, fiba, freq, patch_trigger, BELT, HCB')
    parser.add_argument('--poison_ratio', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, default=7)
    # ---------------------------- For Poisoned Image Smoothing ------------------------
    parser.add_argument('--smooth_type', type=str, default="no_smooth",
                        help='gaussian, wiener, BM3D, no_smooth')

    return parser.parse_args()

args = args_parser()
criterion = nn.CrossEntropyLoss()
criterion.to(args.device)


if args.data == "imagenet_full":
    trainset = torch.load('./data/train_sub_imagenet.pt')
    testset = torch.load('./data/test_sub_imagenet.pt')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    print("a pre-selected imagenet full dataset (10 classes) has been loaded!")
else:
    # load data for train
    train_dataset, test_dataset = load_dataset(args)
    clean_train_dataset = copy.deepcopy(train_dataset)
    clean_test_dataset = copy.deepcopy(test_dataset)
    print(f"Full train dataset size: {len(train_dataset)}")
    print(f"Full test dataset size: {len(test_dataset)}")
    train_loader, test_loader = load_data(args, clean_train_dataset, clean_test_dataset)
print('finish loading dataset')



def test(model, test_loader):
    global test_index, poisoned_test_data_list, poisoned_test_label_list
    model.eval()
    clean_loss_avgmeter = AverageMeter()
    clean_acc_avgmeter = AverageMeter()
    bd_loss_avgmeter = AverageMeter()
    bd_acc_avgmeter = AverageMeter()
    print("1, loading batches from test_loader")
    for batch_idx, (data, label) in enumerate(test_loader):
        #data = smoothing(data, args.smooth_type)
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)
        batch_acc=(output.argmax(1) == label.view(-1,)).float().sum()

        loss=criterion(output,label.view(-1,))
        clean_loss_avgmeter.update(loss.detach(),data.size(0))
        clean_acc_avgmeter.update(batch_acc.detach(),data.size(0),True)
    print("2, loading batches from test_loader")
    for batch_idx, (data, label) in enumerate(test_loader):
        save_path = f'/home/suser/project/Thesis/saved_dataset/{args.model}_{args.attack_mode}_poisoned_test_batch_{batch_idx}.pt'
        if test_index == 0:
            if os.path.exists(save_path):
                poisoned = torch.load(save_path)
                data = poisoned['data']
                label = poisoned['label']
                print(f"Loaded cached {save_path}")
            else:
                if args.attack_mode == 'square':
                    data, label = square_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'sig':
                    data, label = sig_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'blend':
                    data, label = blend_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'ftrojan':
                    data, label = ftrojan_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'fiba':
                    data, label = fiba_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'patch_trigger':
                    data, label = patch_trigger(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'BELT':
                    data, label = BELT_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'HCB':
                    data, label = HCB_poison(args, data, label, args.target_label, poison_ratio=1.0)
                elif args.attack_mode == 'HCBsmile':
                    data, label = HCB_poison2(args, data, label, args.target_label, poison_ratio=1.0)
                else:
                    raise Exception(f'Error, unknown attack mode{args.attack_mode}')
                torch.save({'data': data, 'label': label}, save_path)

                print(f"Saved {args.model}_{args.attack_mode}_TEST_batch_{batch_idx}.pt")

                # print(f"Saved TEST_batch_{batch_idx}.pt")
                # print("finish poisoning testdata")
        else:
            poisoned = torch.load(save_path)
            data = poisoned['data']
            label = poisoned['label']
        # print(f"Loaded cached {save_path}")
        # data = smoothing(data, args.smooth_type)
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()

        loss = criterion(output, label.view(-1, ))
        bd_loss_avgmeter.update(loss.detach(), data.size(0))
        bd_acc_avgmeter.update(batch_acc.detach(), data.size(0), True)
    test_index += 1
    model.train()

    return clean_loss_avgmeter.avg, clean_acc_avgmeter.avg, bd_loss_avgmeter.avg, bd_acc_avgmeter.avg


def main():
    global train_index, poisoned_data_list, poisoned_label_list
    if args.data == "cifar10":
        args.num_classes = 10
    elif args.data == "gtsrb":
        args.num_classes = 43
    elif args.data == "imagenet":
        args.num_classes = 200
    elif args.data == "celeba":
        args.num_classes = 8
    elif args.data == "mnist":
        args.num_classes = 10
    elif args.data == "svhn":
        args.num_classes = 10
    elif args.data == "imagenet_full":
        args.num_classes = 1000

    if args.data == "imagenet_full":
        if args.model == "resnet":
            # print('load torchvision.models.resnet18 with IMAGENET1K_V1 pretrained weights on imagenet task')
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            # model = torchvision.models.resnet18()
        else:
            raise Exception('error, the model '+args.model+' is not supported for imagenet_full dataset')
    else:
        if args.model == "vgg16":
            model = vgg16_bn(num_classes=args.num_classes)
        elif args.model == "vgg11":
            model = vgg11_bn(num_classes=args.num_classes)
        elif args.model == "preact_resnet":
            model = PreActResNet18(num_classes=args.num_classes)
        elif args.model == "cnn_mnist":
            model = CNN_MNIST()
        elif args.model == "resnet":
            from models.resnet import ResNet18
            #model = ResNet18(num_classes=args.num_classes)
            # from models.resnet_cifar10 import ResNet18
            model = ResNet18(num_classes=args.num_classes)
        elif args.model == "googlenet":
            model = GoogLeNet()
        else:
            raise Exception('unknown model name')

    model = model.to(args.device)

    save_name = "train_attack_" + args.attack_mode + "_" + args.model + "_" + args.data + "_" + str(args.poison_ratio)
    logging.basicConfig(filename=args.log_dir + save_name + '.txt', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.FileHandler(args.log_dir + save_name + '.txt', mode='w+')

    loss_avgmeter = AverageMeter()
    acc_avgmeter = AverageMeter()

    counter = 0
    best_acc = 0
    best_bd_acc = 0
    clean_test_loss_min = 0
    lr = args.lr
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_index += 1
        model.train()

        if counter / 10 == 1:
            counter = 0
            lr = lr * 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # print("1, loading batches from train_loader")
        for batch_idx, (data, label) in enumerate(train_loader):
            save_path = f'/home/suser/project/Thesis/saved_dataset/{args.model}_{args.attack_mode}_poisoned_batch_{batch_idx}.pt'
            if os.path.exists(save_path):
                poisoned = torch.load(save_path)
                data = poisoned['data']
                label = poisoned['label']
                # print("2, loaded batches from train_loader")
                # print(f"Loaded cached {save_path}")
            else:
                if train_index == 1:
                    extra_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),  # preprocess
                        #transforms.RandomCrop((32, 32), padding=4),  # preprocess
                        transforms.RandomCrop((64, 64), padding=4),  # preprocess
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
                    ])
                    data = torch.stack([extra_transform(img) for img in data])

                    if args.attack_mode == 'square':
                        data, label = square_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'sig':
                        data, label = sig_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'blend':
                        data, label = blend_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'ftrojan':
                        data, label = ftrojan_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'patch_trigger':
                        data, label = patch_trigger(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'BELT':
                        data, label = BELT_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'HCB':
                        data, label = HCB_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'fiba':
                        data, label = fiba_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    elif args.attack_mode == 'HCBsmile':
                        data, label = HCB_poison2(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
                    else:
                        raise Exception(f'Error, unknown attack mode{args.attack_mode}')
                    # poisoned_data_list.append(data)
                    # poisoned_label_list.append(label)
                    
                    torch.save({'data': data, 'label': label}, save_path)

                    print(f"Saved {args.model}_{args.attack_mode}_poisoned_batch_{batch_idx}.pt")

                    # print("finish poisoning traindata")
                else:
                    # data = poisoned_data_list[batch_idx]
                    # label = poisoned_label_list[batch_idx]
                    poisoned = torch.load(save_path)
                    data = poisoned['data']
                    label = poisoned['label']
            # print(f"smoothing data with {args.smooth_type} type")
            # data = smoothing(data, args.smooth_type)
            # print(f"FINISH smoothing data with {args.smooth_type} type")
            data = data.to(args.device)
            label = label.to(args.device)
            output = model(data)
            batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()
            optimizer.zero_grad()
            loss = criterion(output,label.view(-1,))
            loss_avgmeter.update(loss.detach(), data.size(0))
            acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

            loss.backward()
            optimizer.step()

        time_elapsed = time.time() - start_time

        train_avg_loss = loss_avgmeter.avg
        train_avg_acc = acc_avgmeter.avg

        clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss, bd_test_avg_acc = test(model, test_loader)

        print("""{}/{}, Avg Train Loss:{:.6f}, Avg Train Acc:{:.4f}, 
        Avg Test Loss Clean:{:.6f}, Avg Test Acc Clean:{:.4f}, 
  cannot      Avg Test Loss backdoor:{:.6f}, Avg Test Acc backdoor:{:.4f}, Best Acc:\033[91m{:.4f} \033[0m,
        Best Backdoor Acc:\033[91m{:.4f} \033[0m""".\
                format(epoch,args.epochs, train_avg_loss, train_avg_acc,
                       clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss,
                       bd_test_avg_acc,best_acc,best_bd_acc))
        logging.info(f'Epoch {epoch + 1}/{args.epochs}, Train_Loss: {train_avg_loss}, Train_Accuracy: {train_avg_acc}, '
                     f'Clean_Test_Loss: {clean_test_avg_loss}, Clean_Test_Accuracy: {clean_test_avg_acc}, '
                     f'Bd_Test_Loss: {bd_test_avg_loss}, Bd_Test_Accuracy: {bd_test_avg_acc}')
        print('Elapsed Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Current learning rate:{:.6f}'.format(lr))
        loss_avgmeter.reset()
        acc_avgmeter.reset()

        if best_bd_acc <= bd_test_avg_acc:
            best_bd_acc = bd_test_avg_acc

        if best_acc <= clean_test_avg_acc:
            best_acc = clean_test_avg_acc

        if clean_test_avg_loss < clean_test_loss_min:
            clean_test_loss_min = clean_test_avg_loss
            counter = 0
        else:
            counter += 1

        # # save model
        # path = os.path.join(args.model_dir, f'{save_name}.pt')
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # torch.save(model, path)

# def test(model, test_loader):
#     global test_index, poisoned_test_data_list, poisoned_test_label_list
#     model.eval()
#     clean_loss_avgmeter = AverageMeter()
#     clean_acc_avgmeter = AverageMeter()
#     bd_loss_avgmeter = AverageMeter()
#     bd_acc_avgmeter = AverageMeter()

#     for batch_idx, (data, label) in enumerate(test_loader):
#         #data = smoothing(data, args.smooth_type)
#         data = data.to(args.device)
#         label = label.to(args.device)
#         output = model(data)
#         batch_acc=(output.argmax(1) == label.view(-1,)).float().sum()

#         loss=criterion(output,label.view(-1,))
#         clean_loss_avgmeter.update(loss.detach(),data.size(0))
#         clean_acc_avgmeter.update(batch_acc.detach(),data.size(0),True)

#     for batch_idx, (data, label) in enumerate(test_loader):
#         if test_index == 0:
#             if args.attack_mode == 'square':
#                 data, label = square_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'sig':
#                 data, label = sig_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'blend':
#                 data, label = blend_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'ftrojan':
#                 data, label = ftrojan_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'fiba':
#                 data, label = fiba_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'patch_trigger':
#                 data, label = patch_trigger(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'BELT':
#                 data, label = BELT_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'HCB':
#                 data, label = HCB_poison(args, data, label, args.target_label, poison_ratio=1.0)
#             elif args.attack_mode == 'HCBsmile':
#                 data, label = HCB_poison2(args, data, label, args.target_label, poison_ratio=1.0)
#             else:
#                 raise Exception(f'Error, unknown attack mode{args.attack_mode}')
#             poisoned_test_data_list.append(copy.deepcopy(data))
#             poisoned_test_label_list.append(copy.deepcopy(label))
#                 # print("finish poisoning testdata")
#         else:
#             data = poisoned_test_data_list[batch_idx]
#             label = poisoned_test_label_list[batch_idx]

#         data = data.to(args.device)
#         label = label.to(args.device)
#         output = model(data)

#         batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()

#         loss = criterion(output, label.view(-1, ))
#         bd_loss_avgmeter.update(loss.detach(), data.size(0))
#         bd_acc_avgmeter.update(batch_acc.detach(), data.size(0), True)
#     test_index += 1
#     model.train()

#     return clean_loss_avgmeter.avg, clean_acc_avgmeter.avg, bd_loss_avgmeter.avg, bd_acc_avgmeter.avg



# def main():
#     global train_index, poisoned_data_list, poisoned_label_list
#     if args.data == "cifar10":
#         args.num_classes = 10
#     elif args.data == "gtsrb":
#         args.num_classes = 43
#     elif args.data == "imagenet":
#         args.num_classes = 200
#     elif args.data == "celeba":
#         args.num_classes = 8
#     elif args.data == "mnist":
#         args.num_classes = 10
#     elif args.data == "svhn":
#         args.num_classes = 10
#     elif args.data == "imagenet_full":
#         args.num_classes = 1000

#     if args.data == "imagenet_full":
#         if args.model == "resnet":
#             print('load torchvision.models.resnet18 with IMAGENET1K_V1 pretrained weights on imagenet task')
#             model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
#             # model = torchvision.models.resnet18()
#         else:
#             raise Exception('error, the model '+args.model+' is not supported for imagenet_full dataset')
#     else:
#         if args.model == "vgg16":
#             model = vgg16_bn(num_classes=args.num_classes)
#         elif args.model == "vgg11":
#             model = vgg11_bn(num_classes=args.num_classes)
#         elif args.model == "preact_resnet":
#             model = PreActResNet18(num_classes=args.num_classes)
#         elif args.model == "cnn_mnist":
#             model = CNN_MNIST()
#         elif args.model == "resnet":
#             from models.resnet import ResNet18
#             #model = ResNet18(num_classes=args.num_classes)
#             # from models.resnet_cifar10 import ResNet18
#             model = ResNet18(num_classes=args.num_classes)
#         elif args.model == "googlenet":
#             model = GoogLeNet()
#         else:
#             raise Exception('unknown model name')

#     model = model.to(args.device)

#     save_name = "train_attack_" + args.attack_mode + "_" + args.model + "_" + args.data + "_" + str(args.poison_ratio)
#     logging.basicConfig(filename=args.log_dir + save_name + '.txt', level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s')
#     logging.FileHandler(args.log_dir + save_name + '.txt', mode='w+')

#     loss_avgmeter = AverageMeter()
#     acc_avgmeter = AverageMeter()

#     counter = 0
#     best_acc = 0
#     best_bd_acc = 0
#     clean_test_loss_min = 0
#     lr = args.lr
#     start_time = time.time()
    
#     for epoch in range(args.epochs):
#         train_index += 1
#         model.train()

#         if counter / 10 == 1:
#             counter = 0
#             lr = lr * 0.5
#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

#         for batch_idx, (data, label) in enumerate(train_loader):
#             if train_index == 1:
#                 extra_transform = transforms.Compose([
#                     transforms.RandomHorizontalFlip(p=0.5),  # preprocess
#                     #transforms.RandomCrop((32, 32), padding=4),  # preprocess
#                     transforms.RandomCrop((64, 64), padding=4),  # preprocess
#                     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # preprocess
#                 ])
#                 data = torch.stack([extra_transform(img) for img in data])

#                 if args.attack_mode == 'square':
#                     data, label = square_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'sig':
#                     data, label = sig_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'blend':
#                     data, label = blend_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'ftrojan':
#                     data, label = ftrojan_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'patch_trigger':
#                     data, label = patch_trigger(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'BELT':
#                     data, label = BELT_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'HCB':
#                     data, label = HCB_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 elif args.attack_mode == 'HCBsmile':
#                     data, label = HCB_poison2(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
#                 else:
#                     raise Exception(f'Error, unknown attack mode{args.attack_mode}')
#                 poisoned_data_list.append(data)
#                 poisoned_label_list.append(label)
                
#                 # torch.save({'data': data, 'label': label}, save_path)
#                 print(f"Saved poisoned_batch_{batch_idx}.pt")
#                 # print("finish poisoning traindata")
#             else:
#                 data = poisoned_data_list[batch_idx]
#                 label = poisoned_label_list[batch_idx]

       
#             data = data.to(args.device)
#             label = label.to(args.device)
#             output = model(data)
#             batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()
#             optimizer.zero_grad()
#             loss = criterion(output,label.view(-1,))
#             loss_avgmeter.update(loss.detach(), data.size(0))
#             acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

#             loss.backward()
#             optimizer.step()

#         time_elapsed = time.time() - start_time

#         train_avg_loss = loss_avgmeter.avg
#         train_avg_acc = acc_avgmeter.avg

#         clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss, bd_test_avg_acc = test(model, test_loader)

#         print("""{}/{}, Avg Train Loss:{:.6f}, Avg Train Acc:{:.4f}, 
#         Avg Test Loss Clean:{:.6f}, Avg Test Acc Clean:{:.4f}, 
#   cannot      Avg Test Loss backdoor:{:.6f}, Avg Test Acc backdoor:{:.4f}, Best Acc:\033[91m{:.4f} \033[0m,
#         Best Backdoor Acc:\033[91m{:.4f} \033[0m""".\
#                 format(epoch,args.epochs, train_avg_loss, train_avg_acc,
#                        clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss,
#                        bd_test_avg_acc,best_acc,best_bd_acc))
#         logging.info(f'Epoch {epoch + 1}/{args.epochs}, Train_Loss: {train_avg_loss}, Train_Accuracy: {train_avg_acc}, '
#                      f'Clean_Test_Loss: {clean_test_avg_loss}, Clean_Test_Accuracy: {clean_test_avg_acc}, '
#                      f'Bd_Test_Loss: {bd_test_avg_loss}, Bd_Test_Accuracy: {bd_test_avg_acc}')
#         print('Elapsed Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#         print('Current learning rate:{:.6f}'.format(lr))
#         loss_avgmeter.reset()
#         acc_avgmeter.reset()

#         if best_bd_acc <= bd_test_avg_acc:
#             best_bd_acc = bd_test_avg_acc

#         if best_acc <= clean_test_avg_acc:
#             best_acc = clean_test_avg_acc

#         if clean_test_avg_loss < clean_test_loss_min:
#             clean_test_loss_min = clean_test_avg_loss
#             counter = 0
#         else:
#             counter += 1

#         # save model
#         # path = os.path.join(args.model_dir, f'{save_name}.pt')
#         # os.makedirs(os.path.dirname(path), exist_ok=True)
#         # torch.save(model, path)
#         # if bd_test_avg_acc > 0.95 and clean_test_avg_acc > 0.70:
#         #     epoch_model_path = os.path.join(args.model_dir, f'{save_name}_epoch{epoch+1}.pt')
#         #     os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
#         #     torch.save(model, epoch_model_path)
#         # if bd_test_avg_acc > 0.95 and clean_test_avg_acc > 0.7:
        model_name = f'{save_name}_{args.attack_mode}_{args.smooth_type}_epoch{epoch+1}.pt'
        epoch_model_path = os.path.join(args.model_dir, model_name)
        os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
        torch.save(model, epoch_model_path)

if __name__ == '__main__':
    main()
