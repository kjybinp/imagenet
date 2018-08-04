import os
import argparse
import shutil


def get_jpg_namelist(path):
    ls = os.listdir(path)
    ls_jpg = []
    for file in ls:
        if ('.png' in file):
            ls_jpg.append(file)
    assert len(ls_jpg) > 0, 'jpgファイルが１つも見つかりませんでした。'
    return ls_jpg

def main():
    parser = argparse.ArgumentParser(description='フォルダから分類用のラベルを作成')
    parser.add_argument('--dir', help='画像が置いているディレクトリ',
                        default='C:\\Users\\yawata\\Desktop\\workspace\\src\chainer\\examples\\imagenet\\101_ObjectCategories')
    args = parser.parse_args()

    # labels
    images_dir = args.dir
    print(images_dir + 'から画像を探します。')
    labels = os.listdir(images_dir)

    # make directries
    pwd = os.getcwd()
    if(not('imgaes' in os.listdir(pwd))):
        os.mkdir('images')

    # copy images and make train.txt
    imageDir = pwd + "\\images"
    train = open('train.txt', 'w')
    test = open('test.txt', 'w')
    labelsTxt = open('labels.txt', 'w')

    classNo = 0
    cnt = 0
    # label = labels[classNo]
    for label in labels:
        print('class ' + label + 'を処理中')
        workdir = images_dir + "/" + label
        images = get_jpg_namelist(workdir)
        labelsTxt.write(label + "\n")
        startCnt = cnt
        length = len(images)
        for image in images:
            imagepath = imageDir + "\\image%07d" % cnt + ".png"
            shutil.copy(workdir + '/' + image, imagepath)
            if cnt - startCnt < length * 0.75:
                train.write(imagepath + " %d\n" % classNo)
            else:
                test.write(imagepath + " %d\n" % classNo)
            cnt += 1

        classNo += 1

    train.close()
    test.close()
    labelsTxt.close()


if __name__ == '__main__':
    main()