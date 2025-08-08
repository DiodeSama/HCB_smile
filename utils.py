import numpy
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import math
import copy
import lpips
import pytorch_ssim
import torchvision.transforms as T

def preprocessor(img, process_type, deepcopy = True, return_type= 'ndarray', resize_pair=None):
    """
    Args:
        img: can be a ndarray or a tensor
        process_type: resize randomhorizontalflip randomcrop randomrotation
        deepcopy: whether copy the image before preprocessing (True by default)
        resize_pair: (w,h) required by resize and randomcrop
    Returns:
        processed image in numpy format
    """

    # preprocessor = T.Compose([T.Resize((24, 24))])
    # preprocessor = T.Compose([T.Resize((24, 24)), T.ToTensor()])

    if deepcopy:
        img = copy.deepcopy(img)
    if isinstance(img, numpy.ndarray):
        if not deepcopy:
            img = copy.deepcopy(img)
        img = torch.tensor(img)
    if len(img) == 4:
        img = img.unsqueeze(0)
    if img.shape[0] == img.shape[1]:
        img = img.transpose(2,0,1)

    if process_type.lower() == 'resize':
        preprocessor = T.Compose([T.Resize(resize_pair)])
    elif process_type.lower() == 'randomhorizontalflip':
        preprocessor = T.Compose([T.RandomHorizontalFlip(p=0.5)])
    elif process_type.lower() == 'randomcrop':
        preprocessor = T.Compose([T.RandomCrop(resize_pair, padding=4)])
    elif process_type.lower() == 'randomrotation':
        preprocessor = T.Compose([T.RandomRotation(10)])
    else:
        raise Exception('Unknown transformation type in utils.py->preprocessor:'+str(process_type))
    img = preprocessor(img)
    return img.numpy()


def ssim(img1, img2):

    if len(img1.shape) == 4 or len(img2.shape) == 4:
        raise Exception('error, only single picture is accepted')

    if isinstance(img1, numpy.ndarray):
        img1 = torch.tensor(img1)
    if isinstance(img2, numpy.ndarray):
        img2 = torch.tensor(img2)

    if img1.shape[0] == img1.shape[1]:
        torch.permute(img1,(2,0,1))
    if img2.shape[0] == img2.shape[1]:
        torch.permute(img2,(2,0,1))

    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)

    return pytorch_ssim.ssim(img1, img2).item()

def Lpips(img1, img2):

    if len(img1.shape) == 4 or len(img2.shape) == 4:
        raise Exception('error, only single picture is accepted')

    if isinstance(img1, numpy.ndarray):
        img1 = torch.tensor(img1)
    if isinstance(img2, numpy.ndarray):
        img2 = torch.tensor(img2)

    if img1.shape[0] == img1.shape[1]:
        torch.permute(img1,(2,0,1))
    if img2.shape[0] == img2.shape[1]:
        torch.permute(img2,(2,0,1))

    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores  #

    return loss_fn_alex(img1, img2).item()


def psnr(img1, img2):

    if len(img1.shape) == 4 or len(img2.shape) == 4:
        raise Exception('error, only single picture is accepted')

    if isinstance(img1, numpy.ndarray):
        img1 = torch.tensor(img1)
    if isinstance(img2, numpy.ndarray):
        img2 = torch.tensor(img2)

    mse = torch.mean((img1 - img2) ** 2)

    max_value = 1.0

    if mse == 0:
        return 100
    else:
        return 20 * math.log10(max_value / (math.sqrt(mse)))

def LpDIF(img1, img2, p=1): # img1 and img2 can be either numpy array or tensor matrix

    if len(img1.shape) == 4 or len(img2.shape) == 4:
        raise Exception('error, only single picture is accepted')

    if img1.shape!= img2.shape:
        raise Exception('error, dim of img1 and img2 is different')

    if isinstance(img1, torch.Tensor):
        img1 = img1.numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.numpy()

    diff = img1.flatten() - img2.flatten()
    return np.linalg.norm(diff, ord=p)  # L2 norm should below 1.5


def obtain_nprmalized_reciprocal_matrix_of_DCT_strength(DCT_mat):
    reciprocal_normalized_strength_matrix = np.zeros(DCT_mat.shape)
    for channel_idx in range(DCT_mat.shape[0]):
        max_strength = np.max(DCT_mat[channel_idx])
        min_strength = np.min(DCT_mat[channel_idx])
        diff = max_strength - min_strength
        if diff <= 0.0000001:
            raise Exception('error, diff=0 in DCT matrix, which seems to be strange')
        for j in range(DCT_mat.shape[1]):
            for k in range(DCT_mat.shape[2]):
                reciprocal_normalized_strength_matrix[channel_idx][j][k] = (DCT_mat[channel_idx][j][k]-min_strength)/diff
        reciprocal_normalized_strength_matrix[channel_idx][reciprocal_normalized_strength_matrix[channel_idx]==0] = np.std(reciprocal_normalized_strength_matrix[channel_idx])
        reciprocal_normalized_strength_matrix[channel_idx] = 1/reciprocal_normalized_strength_matrix[channel_idx]
    return reciprocal_normalized_strength_matrix

def eval_stealthiness_robustness_for_v4(DCT_mat, vars, mask, decay_func):

    reciprocal_normalized_strength_matrix = obtain_nprmalized_reciprocal_matrix_of_DCT_strength(DCT_mat)

    perturb_mat = np.zeros_like(DCT_mat)
    seq = 0
    for i in range(DCT_mat.shape[0]):
        for j in range(DCT_mat.shape[1]):
            for k in range(DCT_mat.shape[2]):
                if mask[j][k] != 0:
                    perturb_mat[i][j][k] = vars[seq] / 255.
                    seq += 1

    sum_perturb_mul_reciprocal_normalized_strength = 0
    for i in range(DCT_mat.shape[0]):
        for j in range(DCT_mat.shape[1]):
            for k in range(DCT_mat.shape[2]):
                sum_perturb_mul_reciprocal_normalized_strength += decay_func(reciprocal_normalized_strength_matrix[i][j][k]) * perturb_mat[i][j][k]
    return sum_perturb_mul_reciprocal_normalized_strength

def eval_stalthiness_robustness(DCT_mat, vars, decay_func, vars_reshape_dim=None, patch_loc=None):

    reciprocal_normalized_strength_matrix = obtain_nprmalized_reciprocal_matrix_of_DCT_strength(DCT_mat)

    if vars_reshape_dim is None:
        vars_2_mat = np.reshape([x / 255 for x in vars], DCT_mat.shape)
    else:
        vars_2_mat = np.reshape([x / 255 for x in vars], vars_reshape_dim)
        vars_2_mat_expand = np.zeros_like(DCT_mat)
        for i in range(vars_reshape_dim[0]):
            for j in range(vars_reshape_dim[1]):
                for k in range(vars_reshape_dim[2]):
                    vars_2_mat_expand[i][j + patch_loc[0]][k + patch_loc[1]] = vars_2_mat[i][j][k]
        vars_2_mat = vars_2_mat_expand


    sum_perturb_mul_reciprocal_normalized_strength = 0
    for i in range(DCT_mat.shape[0]):
        for j in range(DCT_mat.shape[1]):
            for k in range(DCT_mat.shape[2]):
                sum_perturb_mul_reciprocal_normalized_strength += decay_func(reciprocal_normalized_strength_matrix[i][j][k]) * vars_2_mat[i][j][k]
    return sum_perturb_mul_reciprocal_normalized_strength



def gaussian_spatial_domain_smooth(image_ori, sig=1, deep_copy=True):
    # Originated from: https://github.com/YiZeng623/frequency-backdoor/blob/main/Sec5_Smooth_Trigger/gauss_smooth.py

    if deep_copy:
        image = copy.deepcopy(image_ori)
    else:
        image = image_ori

    # the input image should be a tensor with the order of dimension: channel*W*H
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if image.shape[0] != image.shape[1]: # in this case channel is located in the last dimension
        image = image.transpose(1, 2, 0)

    #the required image should have dimension: H*W*C, i.e., the dim of Channel should locate in the last dimension

    size_denom = 5.
    sigma = sig * size_denom
    kernel_size = sigma
    mgrid = np.arange(kernel_size, dtype=np.float32)
    mean = (kernel_size - 1.) / 2.
    mgrid = mgrid - mean
    mgrid = mgrid * size_denom
    kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
             np.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)
    kernel = kernel / np.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernelx = np.tile(np.reshape(kernel, (1, 1, int(kernel_size), 1)), (3, 1, 1, 1))
    kernely = np.tile(np.reshape(kernel, (1, 1, 1, int(kernel_size))), (3, 1, 1, 1))

    padd0 = int(kernel_size // 2)
    evenorodd = int(1 - kernel_size % 2)

    pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.)
    in_put = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32), (2, 0, 1)), axis=0))
    output = pad(in_put)

    weightx = torch.from_numpy(kernelx)
    weighty = torch.from_numpy(kernely)
    conv = torch.nn.functional.conv2d
    output = conv(output, weightx, groups=3)
    output = conv(output, weighty, groups=3)

    output = output[0].numpy().transpose(1,2,0) #finally the output should have dim C*H*W

    return output



def poison_data_with_normal_trigger(data, target, target_label, device = 'cpu', poison_frac=0.2, agent_no=-1):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)

    target_tensor = []
    poison_number = math.floor(len(target) * poison_frac)
    trigger_value = 0
    pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
                    [[0, 6], [0, 7], [0, 8], [0, 9]],
                    [[3, 0], [3, 1], [3, 2], [3, 3]],
                    [[3, 6], [3, 7], [3, 8], [3, 9]]]
    if agent_no == -1:
        for index in range(0, poison_number):
            target[index] = target_label
            for channel in range(3):
                for i in range(len(pattern_type)):
                    for j in range(len(pattern_type[i])):
                        pos = pattern_type[i][j]
                        data[index][channel][pos[0]][pos[1]] = trigger_value
    else:
        for index in range(poison_number):
            target[index] = target_label
            for channel in range(3):
                for j in range(len(pattern_type[agent_no])):
                    pos = pattern_type[agent_no][j]
                    data[index][channel][pos[0]][pos[1]] = trigger_value

    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    target = target[random_perm]

    return data.to(device=device), target.to(device=device)
def test_mali_normal_trigger(model, test_loader, target_label, device = 'cpu', verbose=True):
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        # deep copy of data and target is implemented in poison_data_with_normal_trigger
        data, target = poison_data_with_normal_trigger(data, target, target_label, poison_frac=1.0)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    if verbose:
        print('mali accuracy  = {}'.format(acc))
    return acc


def test_model(model, test_loader, device = 'cpu', verbose=True):
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


def if_reach_target_label(poisoned_img, model, target=7, device='cpu', targeted=True):

    sample = torch.tensor(poisoned_img).unsqueeze(0).to(device)
    output = model(sample)
    output = output.detach().cpu()
    _, pred_labels = torch.max(output, 1)
    pred_labels = pred_labels.view(-1)
    if targeted:
        if pred_labels.item() == target:
            return True
        else:
            return False
    else:
        return pred_labels.item()


def clip_pixel_in_spatial_space(img):
    if isinstance(img, torch.Tensor):
        raise Exception('please dont clip image in tensor format')
    adjust_channel_order = False
    copy_img = copy.deepcopy(img)
    if copy_img.shape[0] == copy_img.shape[1]:
        copy_img = copy_img.transpose(2, 0, 1)
    copy_img[0][0][0] = -12
    copy_img[0][0][1] = -12
    copy_img[0][0][2] = -12
    copy_img[1][0][0] = -12
    copy_img[1][0][1] = -12
    copy_img[1][0][2] = -12
    copy_img[2][0][0] = -12
    copy_img[2][0][1] = -12
    copy_img[2][0][2] = -12

    copy_img[copy_img < 0.0] = 0.0
    if sum(copy_img.flatten())/copy_img.size <= 1.0:
        copy_img[copy_img > 1.0] = 1.0
    else:
        copy_img[copy_img > 255] = 255
    return copy_img

def save_pic_ablation(img_mat, file_name):

    if (isinstance(img_mat, torch.Tensor)):
        img_mat_np = img_mat.numpy()
    else:
        img_mat_np = copy.deepcopy(img_mat)
    if img_mat_np.shape[0] == 3:
        img_mat_np = img_mat_np.transpose(1, 2, 0)
    plt.axis('off')
    plt.xticks([])

    plt.imshow(img_mat_np)
    plt.savefig('./ablation/visual_inspect_images/'+file_name,bbox_inches='tight', pad_inches = -0.1)
    #plt.show()

def show_pic(img_mat, title="poisoned image"):

    if(isinstance(img_mat,torch.Tensor)):
        img_mat_np = img_mat.numpy()
    else:
        img_mat_np = copy.deepcopy(img_mat)
    if img_mat_np.shape[0] == 3:
        img_mat_np = img_mat_np.transpose(1,2,0)
    plt.title(title)

    plt.imshow(img_mat_np)
    plt.show()

def RGB2YUV(x_rgb):
    img = cv2.cvtColor(x_rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    return img

def YUV2RGB(x_yuv):
    img = cv2.cvtColor(x_yuv.astype(np.float32), cv2.COLOR_YCrCb2RGB)
    return img

def convert_01_0255(img):
    need_transpose = False
    if img.shape[0] == 1 and img.shape[0] == 3:
        img = img.transpose(1,2,0)
        need_transpose = True
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if need_transpose:
        img_normalized = img_normalized.transpose(2,0,1)
    return img_normalized

def convert_0255_01(img):
    need_transpose = False
    if img.shape[0] == 1 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        need_transpose = True
    img_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if need_transpose:
        img_normalized = img_normalized.transpose(2, 0, 1)
    return img_normalized


def DCT(pic, window_size, transpose=True):
    # pic: 512*512*3
    # x_train: (idx, w, h, ch)
    if transpose:
        x_train = np.transpose(pic, (2, 0, 1)) #ch, w, h
    else:
        x_train = pic
    #x_dct = np.zeros((pic.shape[2], pic.shape[0], pic.shape[1]), dtype=float)
    x_dct = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=float)
    for ch in range(x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_dct = cv2.dct(x_train[ch][w:w+window_size, h:h+window_size].astype(float))
                x_dct[ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct            # x_dct: (idx, ch, w, h)

def IDCT(x_train, window_size, transpose=True):
    # x_train: (idx, ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=float)
    for ch in range(0, x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_idct = cv2.idct(x_train[ch][w:w+window_size, h:h+window_size].astype(float))
                x_idct[ch][w:w+window_size, h:h+window_size] = sub_idct
    if transpose:
        x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct




#ND-Sort tool
#come from: https://github.com/in3rtial/nondominated/blob/main/sortnd.py#L85
from collections import defaultdict, namedtuple
from typing import List
Individual = namedtuple("Individual", ["index", "fitness"])
class NDSort:
    def isDominated(self, wvalues1: List, wvalues2: List):
        not_equal = False
        for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
            if self_wvalue > other_wvalue:
                return False
            elif self_wvalue < other_wvalue:
                not_equal = True
        return not_equal

    def sortNonDominated(self, individuals):
        map_fit_ind = defaultdict(list)
        for ind in individuals:
            map_fit_ind[ind.fitness].append(ind)
        fits = list(map_fit_ind.keys())

        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)
        dominated_fits = defaultdict(list)

        # Rank first Pareto front
        for i, fit_i in enumerate(fits):
            for fit_j in fits[i + 1:]:
                if self.isDominated(fit_j, fit_i):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif self.isDominated(fit_i, fit_j):
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0:
                current_front.append(fit_i)

        fronts = [[]]
        for fit in current_front:
            fronts[-1].extend(map_fit_ind[fit])
        pareto_sorted = len(fronts[-1])

        # Rank the next front until all individuals are sorted
        N = len(individuals)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

        return fronts

    def sort(self, F):
        to_sort = []
        for i in range(0,len(F)):
            objs = tuple(F[i])
            to_sort.append(Individual(i, objs))
        fronts = self.sortNonDominated(to_sort)
        nd_objs = []
        nd_idxs = []
        for front_index, front in enumerate(fronts):
            for idx, elem in front:
                nd_objs.append(list(front[idx].fitness))
                nd_idxs.append(front[idx].index)
        return np.array(nd_objs), nd_idxs

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