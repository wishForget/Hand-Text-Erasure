# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import paddle
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check, get_image_list
from paddleseg.core import predict
import cv2
import numpy as np
import math
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help=
        'The path of image, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./segoutput')

    # augment for prediction
    parser.add_argument(
        '--aug_pred',
        dest='aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window prediction
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to prediction by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help=
        'The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help=
        'The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    # custom color map
    parser.add_argument(
        '--custom_color',
        dest='custom_color',
        nargs='+',
        help=
        'Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int,
        default=None)
    return parser.parse_args()


def get_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    if args.custom_color:
        test_config['custom_color'] = args.custom_color

    return test_config


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]


    pred_saved_dir = save_dir

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im = cv2.imread(im_path)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # # save added image
            # added_saved_dir = os.path.join(save_dir, 'added_prediction')
            # added_image = utils.visualize.visualize(
            #     im_path, pred, color_map, weight=0.5)
            # added_image_path = os.path.join(added_saved_dir, im_file)
            # mkdir(added_image_path)
            # cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            # pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_mask = pred * 255
            pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            # pred_mask.save(pred_saved_path)
            cv2.imwrite(pred_saved_path, pred_mask)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    test_config = get_test_config(cfg, args)
    config_check(cfg, val_dataset=val_dataset)

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)


if __name__ == '__main__':
    args = parse_args()
    args.cfg = "work/pre_and_submit/cof1.yml"
    args.model_path = "work/pre_and_submit/model1.pdparams"
    args.image_path = r"data/dehw_testB_dataset"
    main(args)

