import os
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from .network import InpaintGenerator, Discriminator
from .network import MaskInpaintGenerator_v5 as MaskInpaintGenerator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, MaskRefineLoss, TVLoss, VGG19
from paddle import summary
from paddle.optimizer.lr import LRScheduler


class BaseModel(nn.Layer):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.MODEL_DIR, name + '_gen_%d.pdparams')
        self.dis_weights_path = os.path.join(config.MODEL_DIR, name + '_dis_%d.pdparams')

        self.pre_gen_weights_path = config.G_MODEL_PATH
        self.pre_dis_weights_path = config.D_MODEL_PATH

    def load(self):
        if self.pre_gen_weights_path is not None:
            print('Loading %s generator...' % self.name)
            assert os.path.exists(self.pre_gen_weights_path) == True
            # assert 'gen' in os.path.basename(self.pre_gen_weights_path)
            # assert self.name in os.path.basename(self.pre_gen_weights_path)

            data = paddle.load(self.pre_gen_weights_path)
            self.generator.set_dict(data['generator'])
            # self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and self.pre_dis_weights_path is not None:
            print('Loading %s discriminator...' % self.name)
            assert os.path.exists(self.pre_dis_weights_path) == True
            assert 'dis' in os.path.basename(self.pre_dis_weights_path)
            assert self.name in os.path.basename(self.pre_dis_weights_path)

            data = paddle.load(self.pre_dis_weights_path)
            self.discriminator.set_dict(data)
            # self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        paddle.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path % self.iteration)

        paddle.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path % self.iteration)


class MaskInpaintModel(BaseModel):
    def __init__(self, config):
        config.G_MODEL_PATH = config.G_MODEL_PATH
        config.D_MODEL_PATH = config.D_MODEL_PATH
        self.mode = config.MODE
        self._with_style_content_loss = config.WITH_STYLE_CONTENT_LOSS
        self._with_feature_match_loss = config.WITH_FEATURE_MATCH_LOSS
        super(MaskInpaintModel, self).__init__('MaskInpaintModel', config)

        # generator input: [image(3) + mask(1)]
        # discriminator input: (image(3) + mask(1))
        generator = MaskInpaintGenerator(in_channels=4, use_spectral_norm=False)
        discriminator = Discriminator(in_channels=4, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        # summary(generator.cuda(), (4, 256, 256))
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        # maskrefine_loss = MaskRefineLoss(type=config.MASK_REFINE_LOSS)
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()

        self.add_sublayer('generator', generator)
        self.add_sublayer('discriminator', discriminator)
        self.add_sublayer('l1_loss', l1_loss)
        self.add_sublayer('adversarial_loss', adversarial_loss)
        if (self._with_style_content_loss and self.mode != 2):
            self.add_sublayer('vgg', VGG19())
            self.add_sublayer('perceptual_loss', perceptual_loss)
            self.add_sublayer('style_loss', style_loss)

        # self.add_sublayer('maskrefine_loss', maskrefine_loss)

        self.gen_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=float(config.LR), step_size=config.STEP_SIZE,
                                                           gamma=0.1)
        self.dis_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=float(config.LR) * float(config.D2G_LR),
                                                           step_size=config.STEP_SIZE, gamma=0.1)
        self.gen_optimizer = paddle.optimizer.Adam(
            parameters=generator.parameters(),
            learning_rate=self.gen_scheduler,
            beta1=config.BETA1, beta2=config.BETA2
        )

        self.dis_optimizer = paddle.optimizer.Adam(
            parameters=discriminator.parameters(),
            learning_rate=self.dis_scheduler,
            beta1=config.BETA1, beta2=config.BETA2
        )
        if config.G_MODEL_PATH is not None:
            self.load()

    def process(self, images, images_gt, masks, use_gt_mask=False):
        # zero optimizers
        self.gen_optimizer.clear_grad()
        self.dis_optimizer.clear_grad()

        # process outputs
        output_images, pre_output_images, output_masks, gan_image = self(images=images, auxiliary=masks,
                                                                                   mask_gt=masks, image_gt=images_gt)

        # to  def forward(self, images, auxiliary, mask_gt=None, image_gt=None):
        # if use_gt_mask:
        #     output_images, pre_output_images, output_masks = self(images, masks, masks_gt)
        # else:
        #     output_images, pre_output_images, output_masks = self(images, masks)
        # 这一句调用了forward
        # output_images, pre_output_images, output_masks = self(images, masks)
        output_images_detach = output_images.detach()
        pre_output_images_detach = pre_output_images.detach()
        gen_loss = 0
        dis_loss = 0
        pre_dis_loss = 0
        # USE_COMPLETE = False
        USE_COMPLETE = False

        masks_gt = masks
        logs = []
        dis_input_real = paddle.concat((images_gt, masks_gt), axis=1)
        dis_input_fake = paddle.concat((output_images_detach, masks_gt), axis=1)
        pre_dis_input_fake = paddle.concat((pre_output_images_detach, masks_gt), axis=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        pre_dis_fake, pre_dis_fake_feat = self.discriminator(pre_dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        pre_dis_fake_loss = self.adversarial_loss(pre_dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        pre_dis_loss += (dis_real_loss + pre_dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = paddle.concat((output_images, masks_gt), axis=1)
        pre_gen_input_fake = paddle.concat((pre_output_images, masks_gt), axis=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        pre_gen_fake, pre_gen_fake_feat = self.discriminator(pre_gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        pre_gen_gan_loss = self.adversarial_loss(pre_gen_fake, True, False)

        mean_real = paddle.mean(dis_real)
        mean_fake = paddle.mean(dis_fake)
        pre_mean_fake = paddle.mean(pre_dis_fake)

        if USE_COMPLETE:
            outputs4dis_cmp = output_images_detach * masks_gt + images * (1 - masks_gt)
            outputs4gen_cmp = output_images * masks_gt + images * (1 - masks_gt)
            pre_outputs4dis_cmp = pre_output_images_detach * masks_gt + images * (1 - masks_gt)
            pre_outputs4gen_cmp = pre_output_images * masks_gt + images * (1 - masks_gt)

            dis_input_fake_cmp = paddle.concat([outputs4dis_cmp, masks_gt], axis=1)
            gen_input_fake_cmp = paddle.concat([outputs4gen_cmp, masks_gt], axis=1)
            pre_dis_input_fake_cmp = paddle.concat([pre_outputs4dis_cmp, masks_gt], axis=1)
            pre_gen_input_fake_cmp = paddle.concat([pre_outputs4gen_cmp, masks_gt], axis=1)

            dis_fake_cmp, dis_fake_feat_cmp = self.discriminator(dis_input_fake_cmp)
            pre_dis_fake_cmp, pre_dis_fake_feat_cmp = self.discriminator(pre_dis_input_fake_cmp)
            # dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss_cmp = self.adversarial_loss(dis_fake_cmp, False, True)
            pre_dis_fake_loss_cmp = self.adversarial_loss(pre_dis_fake_cmp, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss_cmp) / 2
            pre_dis_loss += (dis_real_loss + pre_dis_fake_loss_cmp) / 2
            dis_loss /= 2
            pre_dis_loss /= 2

            gen_fake_cmp, gen_fake_feat_cmp = self.discriminator(gen_input_fake_cmp)
            pre_gen_fake_cmp, pre_gen_fake_feat_cmp = self.discriminator(pre_gen_input_fake_cmp)
            gen_gan_loss_cmp = self.adversarial_loss(gen_fake_cmp, True, False)
            pre_gen_gan_loss_cmp = self.adversarial_loss(pre_gen_fake_cmp, True, False)
            gen_gan_loss += gen_gan_loss_cmp
            gen_gan_loss /= 2
            pre_gen_gan_loss += pre_gen_gan_loss_cmp
            pre_gen_gan_loss /= 2

            mean_fake = (mean_fake + paddle.mean(dis_fake_cmp)) / 2
            pre_mean_fake = (pre_mean_fake + paddle.mean(pre_dis_fake_cmp)) / 2

        if self._with_feature_match_loss:
            gen_fm_loss = 0
            pre_gen_fm_loss = 0
            for i in range(len(dis_real_feat)):
                gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
                pre_gen_fm_loss += self.l1_loss(pre_gen_fake_feat[i], dis_real_feat[i].detach())
                if USE_COMPLETE:
                    gen_fm_loss += self.l1_loss(gen_fake_feat_cmp[i], dis_real_feat[i].detach())
                    pre_gen_fm_loss += self.l1_loss(pre_gen_fake_feat_cmp[i], dis_real_feat[i].detach())
                    gen_fm_loss /= 2
                    pre_gen_fm_loss /= 2

            gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
            pre_gen_fm_loss = pre_gen_fm_loss * self.config.FM_LOSS_WEIGHT
            gen_loss += gen_fm_loss
            gen_loss += pre_gen_fm_loss
            logs.extend([
                ("l_fm", gen_fm_loss.item()),
                ("l_fm_pre", pre_gen_fm_loss.item()),
            ])

        mean_real_fake_diff = paddle.abs(mean_real - mean_fake)
        pre_mean_real_fake_diff = paddle.abs(mean_real - pre_mean_fake)

        gen_gan_loss = gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
        pre_gen_gan_loss = pre_gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss
        gen_loss += pre_gen_gan_loss
        dis_losses_sum = dis_loss + pre_dis_loss

        # add L1 loss
        l1_loss_weight = self.config.L1_LOSS_WEIGHT
        l1_loss_inner_weight = 10 * masks_gt + 1 - masks_gt
        gen_l1_loss = paddle.mean(self.l1_loss(output_images, images_gt) * l1_loss_inner_weight) * l1_loss_weight
        pre_gen_l1_loss = paddle.mean(
            self.l1_loss(pre_output_images, images_gt) * l1_loss_inner_weight) * l1_loss_weight
        gen_loss += gen_l1_loss
        gen_loss += pre_gen_l1_loss

        if self._with_style_content_loss:
            scl_mask = 10 * masks_gt + 1 - masks_gt
            x_vgg, y_vgg = self.vgg(output_images), self.vgg(images_gt)
            pre_x_vgg = self.vgg(pre_output_images)
            # generator perceptual loss
            gen_content_loss = self.perceptual_loss(x_vgg, y_vgg, scl_mask)
            pre_gen_content_loss = self.perceptual_loss(pre_x_vgg, y_vgg, scl_mask)

            # generator style loss
            gen_style_loss = self.style_loss(x_vgg, y_vgg, scl_mask)
            pre_gen_style_loss = self.style_loss(x_vgg, y_vgg, scl_mask)

            if USE_COMPLETE:
                x_vgg = self.vgg(outputs4gen_cmp)
                pre_x_vgg = self.vgg(pre_outputs4gen_cmp)
                # generator perceptual loss
                gen_content_loss_cmp = self.perceptual_loss(x_vgg, y_vgg, scl_mask)
                pre_gen_content_loss_cmp = self.perceptual_loss(pre_x_vgg, y_vgg, scl_mask)
                gen_content_loss += gen_content_loss_cmp
                pre_gen_content_loss += pre_gen_content_loss_cmp

                # # generator style loss
                gen_style_loss_cmp = self.style_loss(x_vgg, y_vgg, scl_mask)
                pre_gen_style_loss_cmp = self.style_loss(pre_x_vgg, y_vgg, scl_mask)
                gen_style_loss += gen_style_loss_cmp
                pre_gen_style_loss += pre_gen_style_loss_cmp
                #
                gen_content_loss /= 2
                pre_gen_content_loss /= 2
                gen_style_loss /= 2
                pre_gen_style_loss /= 2

            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            pre_gen_content_loss = pre_gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            pre_gen_style_loss = pre_gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_content_loss
            gen_loss += pre_gen_content_loss
            gen_loss += gen_style_loss
            gen_loss += pre_gen_style_loss

            # create logs
            logs.extend([
                ("l_per", gen_content_loss.item()),
                ("l_per_pre", pre_gen_content_loss.item()),
                ("l_sty", gen_style_loss.item()),
                ("l_sty_pre", pre_gen_style_loss.item()),
            ])

        # create logs
        logs.extend([
            ("l_l1", gen_l1_loss.item()),
            ("l_l1_pre", pre_gen_l1_loss.item()),
            ("l_d", dis_loss.item()),
            ("l_d_pre", pre_dis_loss.item()),
            ("l_gan", gen_gan_loss.item()),
            ("l_gan_pre", pre_gen_gan_loss.item()),
        ])

        if True:
            logs.extend([
                ("d_real", mean_real.item()),
                ("d_fake", mean_fake.item()),
                ("d_fake_pre", pre_mean_fake.item()),
                ("d_diff", mean_real_fake_diff.item()),
                ("d_diff_pre", pre_mean_real_fake_diff.item()),
            ])

        # mask_refine_loss = self.maskrefine_loss(output_masks, masks_refine_gt)
        # gen_loss += mask_refine_loss * self.config.MASK_REFINE_LOSS_WEIGHT
        # logs.extend([
        #     ("l_mr1", mask_refine_loss.item()),
        # ])

        logs.extend([
            ("l_G", gen_loss.item()),
            ("l_D", dis_losses_sum.item()),
        ])

        return output_images, pre_output_images, output_masks, gen_loss, dis_losses_sum, logs

    def forward(self, images, auxiliary, mask_gt=None, image_gt=None):
        inputs = paddle.concat((images, auxiliary), axis=1)
        output_images, output_preimages, output_masks, gan_image = self.generator(inputs, mask_gt, image_gt)

        return output_images, output_preimages, output_masks, gan_image

    def backward(self, gen_loss=None, dis_loss=None, dis_retain_graph=False, gen_retain_graph=False):
        self.iteration += 1
        dis_loss.backward(retain_graph=dis_retain_graph)
        # self.dis_optimizer.step()
        gen_loss.backward(retain_graph=gen_retain_graph)
        self.dis_optimizer.step()
        self.gen_optimizer.step()
