# test.py
# !/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':
    loss_func = nn.CrossEntropyLoss()  # 定义损失函数
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )
    if args.gpu:
        net.load_state_dict(torch.load(args.weights))
    else:
        net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    test_time = 10
    best_accuracy = 0  # 最优准确率
    with torch.no_grad():
        # 自己加的本地测试图片
        if 1 == 0:
            from PIL import Image

            # PIL导入本地图片进行识别。
            transform_valid = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ]
            )
            img_path = r"D:\wb_python\special_folder_for_python\work for machine learn\test3.jpg"  # 可能是因为minist数据集的是灰度图像，和白底的图片不一样导致！！！！！！！！！
            test_img_x = Image.open(img_path)
            print("transform前 ", type(test_img_x))
            # 本地加入的测试样本注意resize成与训练样本相同的size，因为要保证全连接层和图片经过卷积层和池化层处理后的尺寸对应得上
            test_img_x = transform_valid(test_img_x)  # 调用ToTensor()将格式从PngImageFile转换成Tensor
            print("transform后 ", type(test_img_x))
            test_img_x = torch.unsqueeze(test_img_x, dim=0)  # 要升2次，符合(N,C,H,W)四维！！
            print("升维度后 ", test_img_x.shape)
            output = net(test_img_x)
            print("shape of image:", test_img_x.shape)
            pred_y = torch.max(output, 1)[1].numpy()
            print("real label is ", pred_y)

        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            if test_time == 0:
                print("\nexit!!!!!!!!!")
                # exit(0)
            test_time -= 1
            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')

            print("shape of image:", image.shape)
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)  # 从大到小并且排序，k=5,dim=1,当k=1时和下面那个max函数一个作用

            # 打印前十个测试结果和真实结果进行对比
            # pred_y = torch.max(output, 1)[1].numpy()
            # print('*' * 10)
            # 记得（B,C,H,W)
            # batch_size == 16的情况下：
            # for i in range(16):
            #     print(image[i].shape)
            #     plt.imshow(image[i][0])
            #     plt.show() # 明显可以看出来这种降维方式是不行的，4通道选其一明显是会成马赛克的！！！也就minist数据集会合适一些
            #
            # print(label, f'real index: {label.shape}')
            # print(pred, f'prediction index1: {pred.shape}')
            ##  这里index1和index2理论上应该是对得上（index2是5个index1中最大的一个)，但是实际上没对上，原因未知
            # print(pred_y, f'prediction index2{pred_y.shape}')
            # print('*' * 10)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print("result is as follows:")
    print(f"The best accuracy is {best_accuracy}")
    print(f"The average accuracy is {correct_1 / len(cifar100_test_loader.dataset)}")
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
