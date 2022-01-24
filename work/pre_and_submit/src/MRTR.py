import os
import random

import numpy as np
from visualdl import LogWriter as SummaryWriter
import paddle
from paddle.io import DataLoader
import paddle.nn as nn
from .dataset import Dataset
from .model import MaskInpaintModel
from .utils import Progbar, create_dir, stitch_images, imsave, output_align
from .metrics import PSNR, EdgeAccuracy
import math
# import paddle.vision.utils as vutils
from .kornia import SSIMLoss as SSIM
from .kornia import gaussian as GaussianBlur2d
from PIL import Image, ImageDraw, ImageFont


class MRTR():

    def __init__(self, config):
        self.config = config
        self.iteration = 0

        self.debug = False
        self.maskpreinpaint_model = MaskInpaintModel(config)
        self.psnr = PSNR(255.0)
        self.ssim = SSIM(5)
        self.mse = paddle.nn.MSELoss()
        self.maskacc = EdgeAccuracy(config.EDGE_THRESHOLD)
        # test mode
        if self.config.MODE == 2 or self.config.MODE == 4:
            self.test_dataset = Dataset(config, config.TEST_DATA, augment=False, training=False)
        else:
            # Create tfboard summary writer
            self.val_info = None
            self.is_best = True
            self.writer = SummaryWriter(self.config.LOG_DIR)
            self.train_dataset = Dataset(config, config.TRAIN_DATA, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_DATA, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.MODEL_DIR, 'samples')
        self.val_path = config.MODEL_DIR
        self.results_path = config.TEST_DIR

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        # self.log_file = os.path.join(config.PATH, 'log.dat')

        self.log_file = os.path.join(config.MODEL_DIR, 'log.dat')

    def load(self):
        self.maskpreinpaint_model.load()

    def save(self):
        self.maskpreinpaint_model.save()

    def train_epoc(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config.BATCH_SIZE,
                                  num_workers=1, drop_last=True,
                                  shuffle=True)
        epoch = 0
        keep_training = True
        max_iteration = int(float(self.config.MAX_ITERS))

        if len(self.train_dataset) == 0:
            print('No training data was provided! Check \'TRAIN_DATA\' value in the configuration file.')
            return

        iteration = 0
        while keep_training:
            epoch += 1
            progbar = Progbar(stateful_metrics=['step'])
            for items in train_loader:
                self.maskpreinpaint_model.train()
                images, images_gt, masks = self.get_inputs(items)
                # train
                # prob = np.minimum(self.config.MASK_SWITCH_RATIO, np.ceil(iteration/self.config.MASK_SWITCH_STEP)/10)
                # prob =0.5
                # use_gt_mask = False if np.random.binomial(1, prob) else True
                if random.random() < 0.01:
                    use_gt_mask = True
                else:
                    use_gt_mask = False
                images_gen, pre_images_gen, masks_gen, gen_loss, dis_loss, logs = \
                    self.maskpreinpaint_model.process(images, images_gt, masks, use_gt_mask=use_gt_mask)
                # masks_cmp = masks_gt if use_gt_mask else masks_gen * masks
                # images_cmp = self.get_complete_preinpaint(masks_cmp, images, images_gen)
                # pre_images_cmp = self.get_complete_preinpaint(masks_cmp, images, pre_images_gen)
                # backward
                self.maskpreinpaint_model.backward(gen_loss, dis_loss)
                iteration = self.maskpreinpaint_model.iteration
                # Tensorboard record: scala
                if iteration % self.config.SAVE_SCALR_AT_STEP == 0:
                    self._write_logs(logs, iteration)

                if iteration % self.config.SAVE_HIST_AT_STEP == 0:
                    for name, value in self.maskpreinpaint_model.named_parameters():
                        self.writer.add_histogram('MaskPreinpaint_weight/' + name, value, iteration)
                        if value.grad is not None:
                            self.writer.add_histogram('MaskPreinpaint_grad/' + name, value.grad.numpy(), iteration)

                if iteration % self.config.SAVE_IMAGE_AT_STEP == 0:
                    '''
                    image = self.get_tensorboard_image([images, images_gt,
                                                        self.gray2rgb(masks), self.gray2rgb(masks_refine_gt),
                                                        self.gray2rgb(masks_gen), self.gray2rgb(masks_cmp),
                                                        pre_images_gen, pre_images_cmp,
                                                        images_gen, images_cmp])
                    '''
                    image = stitch_images(
                        self.postprocess(images.numpy()),
                        self.postprocess(images_gt.numpy()),
                        self.postprocess(images_gen.numpy()),
                        img_per_row=1)
                    image = np.array(image)
                    self.writer.add_image('Train/', image, iteration)

                if iteration % self.config.PRINT_AT_STEP == 0:
                    logs = [("step", str(epoch) + "/" + str(iteration)), ] + logs
                    progbar.print_cur(self.config.PRINT_AT_STEP, values=logs)

                self.iteration = iteration
                self._run_steps_after_train(logs)

                if iteration >= max_iteration:
                    keep_training = False
                    break
            self.maskpreinpaint_model.gen_scheduler.step()
            self.maskpreinpaint_model.dis_scheduler.step()

        self.writer.close()
        print('\nEnd training....')

    def eval_epoc(self):
        if self.config.MODE == 1 or self.config.MODE == 3:
            val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.config.BATCH_SIZE,
                                    drop_last=True, shuffle=True)
            total = len(self.val_dataset)
        else:
            val_loader = DataLoader(dataset=self.test_dataset, batch_size=self.config.BATCH_SIZE,
                                    drop_last=False, shuffle=False)
            total = len(self.test_dataset)
            self.config.N_EVAL = 7

        self.maskpreinpaint_model.eval()

        logs = []
        i_logs = []
        progbar = Progbar(total, stateful_metrics=['it'])
        with paddle.no_grad():
            for _iteration, items in enumerate(val_loader):
                images, images_gt, masks = self.get_inputs(items)  # edge model
                images_gen, pre_images_gen, masks_gen, gen_loss, dis_loss, logs = \
                    self.maskpreinpaint_model.process(images, images_gt, masks,
                                                      use_gt_mask=self.config.EVAL_USE_GT_MASK)
                # masks_cmp = masks_gen * masks
                # images_cmp = self.get_complete_preinpaint(masks_cmp, images, images_gen)
                # pre_images_cmp = self.get_complete_preinpaint(masks_cmp, images, pre_images_gen)

                # mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                # im = Image.composite(im1, im2, mask_blur)

                # metrics
                psnr = self.psnr(self.postprocess(images_gt), self.postprocess(images_gen))
                mae = (paddle.sum(paddle.abs(images_gt - images_gen)) / paddle.sum(images_gt)).astype('float32')
                # psnr_cmp = self.psnr(self.postprocess(images_gt), self.postprocess(images_cmp))
                # mae_cmp = (paddle.sum(paddle.abs(images_gt - images_cmp)) / paddle.sum(images_gt)).astype('float32')
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                # logs.append(('psnr_cmp', psnr_cmp.item()))
                # logs.append(('mae_cmp', mae_cmp.item()))

                ssim = self.ssim(images_gt, images_gen)
                # ssim_cmp = self.ssim(images_gt, images_cmp)
                mse = self.mse(images_gt, images_gen)
                # mse_cmp = self.mse(images_gt, images_cmp)
                logs.append(('ssim', (1 - ssim.item()) * 100))
                # logs.append(('ssim_cmp', (1 - ssim_cmp.item()) * 100))
                logs.append(('mse', mse.item()))
                # logs.append(('mse_cmp', mse_cmp.item()))

                # ssim = self.ssim(images_gt, pre_images_gen)
                # ssim_cmp = self.ssim(images_gt, pre_images_cmp)
                # mse = self.mse(images_gt, pre_images_gen)
                # mse_cmp = self.mse(images_gt, pre_images_cmp)
                # logs.append(('pre_ssim', (1 - ssim.item()) * 100))
                # logs.append(('pre_ssim_cmp', (1 - ssim_cmp.item()) * 100))
                # logs.append(('pre_mse', mse.item()))
                # logs.append(('pre_mse_cmp', mse_cmp.item()))

                # Hack: name of edgeacc
                # mask_precision, mask_recall = self.maskacc(masks_refine_gt * masks, masks_cmp)
                # logs.append(('M_P', mask_precision.item()))
                # logs.append(('M_R', mask_recall.item()))

                logs = logs + i_logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs
                                                                                  if
                                                                                  not x[0].startswith('l_') and not x[
                                                                                      0].startswith('d_')])
                # print(_iteration)

                if _iteration >= self.config.N_EVAL - 1:
                    break

        # Print the average values
        progbar.print_info()

        with open(self.log_file, 'a') as f:
            f.write('%s\n' % str(self.iteration) + progbar.get_info())

        '''
        images = self.get_tensorboard_image([images, images_gt, self.gray2rgb(masks), self.gray2rgb(masks_refine_gt),
                                             self.gray2rgb(masks_gen), self.gray2rgb(masks_cmp),
                                             pre_images_gen, pre_images_cmp,
                                             images_gen, images_cmp])
        '''
        images = stitch_images(
            self.postprocess(images.numpy()),
            self.postprocess(images_gt.numpy()),
            self.postprocess(images_gen.numpy()),
            img_per_row=1)
        images = np.array(images)
        if self.config.MODE == 1 or self.config.MODE == 3:
            # Writing to tfboard summary
            _val_info = {}
            # TODO: ensure following code is correct
            for item, value in progbar.get_average_log_values().items():
                if not item.startswith('l_'):
                    self.writer.add_scalar('Validation/' + item, value, self.iteration)
                    _val_info[item] = value

            # if self.val_info is None:
            #     self.val_info = _val_info
            #     self.is_best = True
            # else:
            #     if self.config.MODEL !=1 :
            #         if _val_info['psnr_cmp'] > self.val_info['psnr_cmp']:
            #             self.val_info = _val_info
            #             self.is_best = True
            #     elif self.config.MODEL == 1 :
            #         # Hack: only looked at mask recall, this might be ugly
            #         if _val_info['M_R'] > self.val_info['M_R']:
            #             self.val_info = _val_info
            #             self.is_best = True
            #     else:
            #         raise
            # get_tensorboard_image(self, img_list)
            # images = vutils.make_grid(images[0], normalize=True, scale_each=True)
            self.writer.add_image('Validation/', images, self.iteration)

    def test_epoc(self):
        self.maskpreinpaint_model.eval()
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        ### !!! FIX TEST
        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gt, masks, masks_gt, masks_refine_gt = self.get_inputs(items)
            index += 1

            output_images, output_pre_images, output_masks = self.maskpreinpaint_model(images, masks)
            output_masks_cmp = output_masks
            output_images_cmp = self.get_complete_preinpaint(output_masks_cmp, images, output_images)
            output_pre_images_cmp = self.get_complete_preinpaint(output_masks_cmp, images, output_pre_images)

            outputs = self.postprocess(output_images)[0]
            outputs_cmp = self.postprocess(output_images_cmp)[0]
            path = os.path.join(self.results_path, name)
            tsplit = name.split('.')
            path_cmp = os.path.join(self.results_path, '%s_cmp.%s' % (tsplit[0], tsplit[1]))
            # print(index, name)

            imsave(outputs, path)
            imsave(outputs_cmp, path_cmp)

            if self.debug:
                input_mask = self.postprocess(masks)[0]
                output_mask = self.postprocess(output_masks)[0]
                images = self.postprocess(images)[0]
                fname, fext = name.split('.')

                imsave(images, os.path.join(self.results_path, fname + '_input.' + fext))
                imsave(input_mask, os.path.join(self.results_path, fname + '_input_mask.' + fext))
                imsave(output_mask, os.path.join(self.results_path, fname + '_output_mask.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.maskpreinpaint_model.eval()

        # model = self.config.MODEL
        items = next(self.sample_iterator)
        with paddle.no_grad():
            images, images_gt, masks = self.get_inputs(items)
            image_per_row = 1
            if self.config.SAMPLE_SIZE <= 6:
                image_per_row = 1

            # edge model
            iteration = self.maskpreinpaint_model.iteration
            output_images, output_pre_images, output_masks, gan_image = self.maskpreinpaint_model(images, masks)
            # output_masks_cmp = output_masks * masks
            # output_images_cmp = self.get_complete_preinpaint(output_masks, images, output_images)
            # output_pre_images_cmp = self.get_complete_preinpaint(output_masks_cmp, images, output_pre_images)

            images = stitch_images(
                self.postprocess(images.numpy()),
                self.postprocess(masks.numpy()),
                self.postprocess(gan_image.numpy()),
                self.postprocess(output_images.numpy()),
                self.postprocess(images_gt.numpy()),
                img_per_row=image_per_row)

            draw = ImageDraw.Draw(images)
            ttfront = ImageFont.truetype('simhei.ttf', 50)  # 字体大小
            fillColor = "#ff0000"
            draw.text((0, 0),
                      "images,masks_gt,output_images,images_gt",
                      fill=fillColor, font=ttfront)  # 文字位置，内容，字体

            # images = stitch_images(
            #     self.postprocess(images.numpy()),
            #     self.postprocess(masks.numpy()),
            #     self.postprocess(masks_refine_gt.numpy()),
            #     self.postprocess(output_masks.numpy()),
            #     self.postprocess(output_masks_cmp.numpy()),
            #     self.postprocess(output_pre_images.numpy()),
            #     self.postprocess(output_pre_images_cmp.numpy()),
            #     self.postprocess(output_images.numpy()),
            #     self.postprocess(output_images_cmp.numpy()),
            #     img_per_row=image_per_row)

            # images = stitch_images(
            #     self.postprocess(images.numpy()),
            #     self.postprocess(masks.numpy()),
            #     self.postprocess(output_masks.numpy()),
            #     self.postprocess(masks_gt.numpy()),
            #     self.postprocess(output_pre_images.numpy()),
            #     self.postprocess(output_images.numpy()),
            #     self.postprocess(images_gt.numpy()),
            #     img_per_row=image_per_row)

            if it is not None:
                iteration = it

        path = os.path.join(self.samples_path)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.transpose([0, 2, 3, 1])
        return img.astype('uint8')

    def gray2rgb(self, img):
        return paddle.concat([img] * 3, axis=1)

    def _run_steps_after_train(self, logs):
        """

        Args:
            logs:

        Returns:

        """
        # log model at checkpoints
        if self.config.LOG_INTERVAL and self.iteration % self.config.LOG_INTERVAL == 0:
            self.log(logs)

        # sample model at checkpoints
        if self.config.SAMPLE_INTERVAL and self.iteration % self.config.SAMPLE_INTERVAL == 0:
            self.sample()

        is_finish_eval = False
        # evaluate model at checkpoints
        if self.config.EVAL_INTERVAL and self.iteration % self.config.EVAL_INTERVAL == 0:
            print('...Eval....\n')
            self.eval_epoc()
            print('...\n')
            is_finish_eval = True

        # # save model at checkpoints
        # if self.config.SAVE_INTERVAL and self.iteration % self.config.SAVE_INTERVAL == 0:
        #     if is_finish_eval:
        #         if self.is_best:
        #             print('...Saving model....')
        #             self.save()
        #     else:
        #         print('...Eval....\n')
        #         self.eval()
        #         print('...\n')
        #         if self.is_best:
        #             print('...Saving model....')
        #             self.save()

        # save model at checkpoints
        if self.config.SAVE_INTERVAL and self.iteration % self.config.SAVE_INTERVAL == 0:
            if is_finish_eval:
                print('...Saving model....')
                self.save()
            else:
                print('...Eval....\n')
                self.eval_epoc()
                print('...\n')
                print('...Saving model....')
                self.save()

    def get_inputs(self, items):
        # if self.config.WITH_EDGE:
        images, images_gt, masks = items
        images = paddle.to_tensor(images, dtype='float32')
        # print(images.shape)
        images_gt = paddle.to_tensor(images_gt, dtype='float32')
        masks = paddle.to_tensor(masks, dtype='float32')

        images = images.transpose([0, 3, 1, 2])
        images_gt = images_gt.transpose([0, 3, 1, 2])
        masks = masks.unsqueeze(-1).transpose([0, 3, 1, 2])
        # print(images.shape,images_gt.shape,masks.shape,masks_refine_gt.shape)
        return images, images_gt, masks

    def get_tensorboard_image(self, img_list):
        col = 5
        images = paddle.concat(img_list, axis=1)
        images = images[0]
        images = images.reshape([len(img_list), -1, 256, 256])
        # images = images.transpose([1,2,0])
        # images = images.reshape([256, 256,3])
        # image = vutils.make_grid(images, nrow=col, normalize=False, scale_each=True)

        # import matplotlib.pyplot as plt
        npgrid = images[0, :, :, :].numpy()
        images = np.transpose(npgrid, (1, 2, 0))
        # plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
        # plt.savefig('out.png')
        return images

    def _write_logs(self, logs, iteration):
        for item, value in logs:
            if item.startswith("l_"):
                self.writer.add_scalar('Train/loss/' + item, value, iteration)
            elif item.startswith("d_"):
                self.writer.add_scalar('Train/diff/' + item, value, iteration)
            else:
                self.writer.add_scalar('Train/' + item, value, iteration)

    def get_auxiliary_with_groundtruth(self, masks, masks_refine_gt, images, images_gt):
        # !!! edge, mask order should be edge, mask. Cant be switched
        auxiliary = paddle.concat([images, masks], axis=1)
        auxiliary_gt = paddle.concat([images_gt, masks_refine_gt], axis=1)
        return auxiliary, auxiliary_gt

    def get_auxiliary(self, masks, images):
        auxiliary = paddle.concat([images, masks], axis=1)
        return auxiliary

    def get_complete_preinpaint(self, mask, input, input_gen):
        output_cmp = (input_gen * mask) + (input * (1 - mask))
        return output_cmp
