import paddle
import paddle.nn as nn


class EdgeAccuracy(nn.Layer):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = paddle.sum(labels.astype('float32'))
        selected = paddle.sum(outputs.astype('float32'))

        if relevant == 0 and selected == 0:
            return paddle.to_tensor(1), paddle.to_tensor(1)

        true_positive = ((outputs == labels) * labels).astype('float32')
        recall = paddle.sum(true_positive) / (relevant + 1e-8)
        precision = paddle.sum(true_positive) / (selected + 1e-8)

        return precision, recall

class PSNR(nn.Layer):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = paddle.log(paddle.to_tensor(10.0))
        max_val = paddle.to_tensor(max_val,dtype='float32')

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * paddle.log(max_val) / base10)

    def __call__(self, a, b):
        mse = paddle.mean((a.astype('float32') - b.astype('float32')) ** 2)

        if mse == 0:
            return paddle.to_tensor(0)

        return self.max_val - 10 * paddle.log(mse) / self.base10
