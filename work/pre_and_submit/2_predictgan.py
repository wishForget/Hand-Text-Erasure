# 脚本运行依赖paddlex
# pip install paddlex
import os.path
import paddle
import paddle.nn as nn
import cv2
import glob
import numpy as np
import argparse
from src.config import Config
from main import set_model_log_output_dir
from src.MRTR import MRTR

tar_size = 512


def extract_red(img: np.ndarray):
    """

    :param img:h*w*c,通道排列为RGB
    :return:单通道图，[0, 255],255为红色区域
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 区间1
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 区间2
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 拼接两个区间
    mask = mask0 + mask1
    # print(np.unique(mask))
    # 保存图片
    mask = np.where(mask > 0, 1, 0)
    return mask.astype("uint8")


def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    """
    解决 cv2.imread 在window平台打开中文路径的问题.
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


def pad_to_2n_power(value, n_power):
    """
    将传入的值补至size的倍数，或者特定形状

    Args:
      value: 传入值
      n_power: 目标数值
    """
    padding = value % n_power
    if padding != 0:
        value += (n_power - padding)

    return value


def tianchong_ref(img_path, tar_size=512):
    image = cv2_imread(img_path)
    # image = cv2.cvtColor(o_img_arr, cv2.COLOR_BGR2RGB)
    im_width = image.shape[0]
    im_height = image.shape[1]
    tar_im_width = pad_to_2n_power(im_width, tar_size)
    tar_im_height = pad_to_2n_power(im_height, tar_size)
    reflect101 = cv2.copyMakeBorder(image, 0, tar_im_width - im_width, 0, tar_im_height - im_height,
                                    cv2.BORDER_REFLECT_101)

    return reflect101


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--config', type=str,
        default='work/pre_and_submit/cof2.yml',
        help='model config file')

    parser.add_argument('--model', type=int,
                        choices=[1, 2, 3, 4],
                        help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default=None)

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default="./scripts/output/20220115-050230/model/MaskInpaintModel_gen_105000.pdparams")

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=1
    )

    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='save_path',
        type=str,
        default='test_result'
    )

    return parser.parse_args()


# image_list = glob.glob("./dataset/img_crop_2150_0.001" + "/*.**g")




def run_step2(image_list, seg_path):
    for n in image_list:
        # 切图
        im = cv2.imread(n)
        im_ry = tianchong_ref(n)
        ifn = os.path.split(n)[1]
        m = ifn.replace(".jpg", ".png")
        m = os.path.join(seg_path, m)
        # mask_red_all = cv2.imread(m)
        mask_red_all = tianchong_ref(m)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_red_all = cv2.dilate(mask_red_all, kernel)  # 膨胀图像
        # mask_red_all = cv2.cvtColor(mask_red_all, cv2.COLOR_GRAY2RGB)
        im_ry = np.where(mask_red_all > 0, 255, im_ry)

        im_width_n = im_ry.shape[0] // tar_size
        im_height_n = im_ry.shape[1] // tar_size
        img_ry = np.zeros((im_width_n * tar_size, im_height_n * tar_size, 3)).astype("uint8")
        img_prry = np.zeros((im_width_n * tar_size, im_height_n * tar_size, 3)).astype("uint8")
        optmask_ry = np.zeros((im_width_n * tar_size, im_height_n * tar_size)).astype("uint8")
        for i in range(im_width_n):
            for j in range(im_height_n):
                img_src = im_ry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size]
                mask_red = mask_red_all[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size][:, :, 0]
                mask_red = np.where(mask_red > 0, 1, 0)
                if np.max(mask_red) > 0:
                    img_src = img_src.astype('float32')
                    img = np.transpose(img_src, (2, 0, 1))
                    img = paddle.to_tensor(img / 255, dtype='float32').unsqueeze(0)
                    mask_red = paddle.to_tensor(mask_red, dtype='float32')
                    mask_red = mask_red.unsqueeze(0).unsqueeze(0)
                    with paddle.no_grad():
                        output_images, output_pre_images, output_masks, image = model_g.maskpreinpaint_model(img,
                                                                                                             mask_red)
                        # image_cmp, pre_image, mask_p, image
                    img_out = output_images
                    img_out = img_out.squeeze(0)
                    img_out = paddle.clip(img_out * 255.0, 0, 255)
                    img_out = paddle.transpose(img_out, [1, 2, 0])
                    img_out = np.uint8(img_out)
                    img_ry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size] = img_out

                    img_out = output_pre_images
                    img_out = img_out.squeeze(0)
                    img_out = paddle.clip(img_out * 255.0, 0, 255)
                    img_out = paddle.transpose(img_out, [1, 2, 0])
                    img_out = np.uint8(img_out)
                    img_prry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size] = img_out

                    img_out = output_masks
                    img_out = img_out.squeeze(0).squeeze(0)
                    img_out = paddle.clip(img_out * 255.0, 0, 255)
                    img_out = np.uint8(img_out)
                    optmask_ry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size] = img_out
                else:
                    output_images, output_pre_images, output_masks = img_src, img_src, mask_red
                    img_ry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size] = output_images
                    img_prry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size] = output_pre_images
                    optmask_ry[i * tar_size:(i + 1) * tar_size, j * tar_size:(j + 1) * tar_size] = output_masks

        mask_all = optmask_ry[0:im.shape[0], 0:im.shape[1]]
        gan_all = img_ry[0:im.shape[0], 0:im.shape[1]]
        img_pr = img_prry[0:im.shape[0], 0:im.shape[1]]
        # optmask = optmask_ry[0:im.shape[0], 0:im.shape[1]]

        # lb = cv2.cvtColor(mask_all, cv2.COLOR_GRAY2RGB)
        # _, Thresh = cv2.threshold(lb, 160, 255, cv2.THRESH_BINARY)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dilated = cv2.dilate(Thresh, kernel)  # 膨胀图像
        # dilated = np.where(dilated > 0, 1, 0).astype("uint8")

        # submit_img = gan_all * dilated + im * (1 - dilated)
        # all_image = np.hstack((im, Thresh, dilated * 255, img_pr, gan_all, submit_img))

# ===============================================================================
        # folder = "./hr_jieguo/submit/"
        # if not os.path.exists(folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(folder)
        # fn = folder + os.path.split(n)[1]
        # cv2.imwrite(fn, gan_all)
        #
        # folder = "./hr_jieguo/img_eras/"
        # if not os.path.exists(folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(folder)
        # fn = folder + os.path.split(n)[1]
        # fn = fn.replace(".jpg", ".png")
        # cv2.imwrite(fn, im_ry[0:im.shape[0], 0:im.shape[1]])
        #
        # folder = "./hr_jieguo/mask_all/"
        # if not os.path.exists(folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(folder)
        # fn = folder + os.path.split(n)[1]
        # fn = fn.replace(".jpg", ".png")
        # cv2.imwrite(fn, mask_all)
# ===========================================================================================
        folder = "./submit/"
        if not os.path.exists(folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder)
        fn = folder + os.path.split(n)[1].replace(".jpg",".png")
        cv2.imwrite(fn, gan_all)
        print(fn + "完成")
        # fnn = "./gan_result/" + os.path.split(n)[1]
        # cv2.imwrite(fnn, all_image)


if __name__ == '__main__':
    image_list = glob.glob("./data/dehw_testB_dataset" + "/*.**g")
    seg_path = "./segoutput"

    args = parse_args()
    config = Config(args.config)
    config.MODE = 2
    config.INPUT_SIZE = 0

    config._dict['WORD_BB_PERCENT_THRESHOLD'] = 0
    config._dict['CHAR_BB_PERCENT_THRESHOLD'] = 0
    config._dict['MASK_CORNER_OFFSET'] = 5
    config._dict['_with_style_content_loss'] = False
    config.G_MODEL_PATH = "work/pre_and_submit/model2.pdparams"
    config = set_model_log_output_dir(config)
    model_g = MRTR(config)
    model_g.maskpreinpaint_model.eval()


    run_step2(image_list, seg_path)
