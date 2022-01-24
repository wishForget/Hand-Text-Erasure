import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from math import exp
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).astype('float32').unsqueeze(0).unsqueeze(0)
    window = paddle.to_tensor(_2D_window.expand([channel, 1, window_size, window_size]))
    return window

# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    img1= paddle.to_tensor(img1)
    img2= paddle.to_tensor(img2)
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq #Var(X)=E[X^2]-E[X]^2
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2  #cov(X,Y)=E[XY]-E[X]E[Y]

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
#类重用窗口
class SSIMLoss(nn.Layer):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()  #对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.l1=paddle.nn.L1Loss(reduction='mean')
    def forward(self, img1, img2):
        l1loss = self.l1(img1,img2)
        (_, channel, _, _) = img1.shape

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return l1loss+0.2*(1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average))

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.shape
    window = create_window(window_size, channel)
    return _ssim(img1, img2, window, window_size, channel, size_average)