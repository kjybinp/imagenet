import argparse
import numpy as np

import chainer
import chainer.functions as F
from PIL import Image

import alex
import googlenet
import googlenetbn
import nin
import resnet50

def predict(net, x):
    h = F.relu(net.conv1(x))
    h = F.local_response_normalization(
        F.max_pooling_2d(h, 3, stride=2), n=5)
    h = F.relu(net.conv2_reduce(h))
    h = F.relu(net.conv2(h))
    h = F.max_pooling_2d(
        F.local_response_normalization(h, n=5), 3, stride=2)

    h = net.inc3a(h)
    h = net.inc3b(h)
    h = F.max_pooling_2d(h, 3, stride=2)
    h = net.inc4a(h)

    l = F.average_pooling_2d(h, 5, stride=3)
    l = F.relu(net.loss1_conv(l))
    l = F.relu(net.loss1_fc1(l))
    l = net.loss1_fc2(l)
    l1 = l

    h = net.inc4b(h)
    h = net.inc4c(h)
    h = net.inc4d(h)

    l = F.average_pooling_2d(h, 5, stride=3)
    l = F.relu(net.loss2_conv(l))
    l = F.relu(net.loss2_fc1(l))
    l = net.loss2_fc2(l)
    l2 = l

    h = net.inc4e(h)
    h = F.max_pooling_2d(h, 3, stride=2)
    h = net.inc5a(h)
    h = net.inc5b(h)

    h = F.average_pooling_2d(h, 7, stride=1)
    h = net.loss3_fc(h)
    return l1 + l2 + h


def main():

    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        'resnext50': resnet50.ResNeXt50,
    }

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--src', '-s', help='input image')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    args = parser.parse_args()

    # Set up a neural network
    model = archs[args.arch]()
    chainer.serializers.load_npz('my_model.npz', model)

    # making networ input from image_file
    mean = np.load('mean.npy')
    mean = mean.astype(np.float16)
    img = Image.open(args.src)
    img = img.resize((model.insize, model.insize))
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float16)
    img -= mean[:,:,:]
    img = img * (1.0 / 255.0)
    x = np.ndarray((1, 3, model.insize, model.insize), dtype=np.float32)
    x[0] = img
    x = chainer.Variable(np.asarray(x))

    #predict
    score = (predict(model, x)).data
    categories = np.loadtxt("labels.txt", str, delimiter="\t")
    result = {}
    for i in range(len(categories)):
        result[categories[i]] = score[0][i]
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)

    #display result
    print('result:' + args.src)
    for i in range(2):
        print(i,result[i][0],result[i][1])

if __name__ == '__main__':
    main()