import os
import argparse
import numpy as np

from PIL import Image


def save_reshape_image(src_path, trg_path, row_size, col_size, remain_mono_img = 1):
    img = Image.open(src_path)
    img_resize = img.resize((row_size, col_size))
    img_dim = (np.array(img_resize)).ndim#白黒画像消去に使用
    if(remain_mono_img or img_dim==3):
        img_resize.save(trg_path)

def get_jpg_namelist(path, extension = 'png'):
    ls = os.listdir(path)
    ls_jpg = []
    for file in ls:
        if (('.' + extension) in file):
            ls_jpg.append(file)
    assert len(ls_jpg) > 0, 'ファイルが１つも見つかりませんでした。ディレクトリもしくは拡張子の確認をお願いします。'
    return ls_jpg

def main():
    parser = argparse.ArgumentParser(description='クラス別階層構造になっている画像を一括でreshape')
    parser.add_argument('--dir', help='画像が置いているディレクトリ',
                        default='image')
    parser.add_argument('--ex', help='画像の拡張子',default='png')
    parser.add_argument('--row_length', default=224)
    parser.add_argument('--col_length', default=224)
    parser.add_argument('--mono', help='mono画像を消したい場合場合は0', default=1)
    args = parser.parse_args()

    images_dir = args.dir
    row_length = args.row_length
    col_length = args.col_length
    ex = args.ex
    mono = args.mono

    # make hierarchal directoris
    main_dir_name = (images_dir.split('\\'))[-1]+'_reshape'
    pwd = os.getcwd()
    if(not(main_dir_name in os.listdir(pwd))):
        os.mkdir(main_dir_name)

    for class_dir_name in os.listdir(images_dir):
        print(class_dir_name)
        input_dir = images_dir + '\\' + class_dir_name
        output_dir = main_dir_name + '\\' + class_dir_name
        if (not (class_dir_name in os.listdir(main_dir_name))):
            os.mkdir(output_dir)
        class_jpgfile_ls = get_jpg_namelist(input_dir, ex)
        for jpg_file in class_jpgfile_ls:
            input_jpg = input_dir + '\\' + jpg_file
            output_jpg = output_dir + '\\' + jpg_file
            save_reshape_image(input_jpg, output_jpg, row_length, col_length, mono)


if __name__ == '__main__':
    main()
