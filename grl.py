from chainer import cuda, function, utils, reporter
from chainer.utils import type_check
import math


class GRL(function.Function):

    def __init__(self, iter_count, upper=1.0):
        self.update_count = iter_count

        # calc_lambda() hyper params
        self.upper = upper
        self.lower = 0.0
        self.alpha = 10.0
        self.max_iter = 10000
        self.current_lmd = self.calc_lambda()
        reporter.report({"GRL/lmd": self.current_lmd})

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    # ---------------------------------------------------------
    # Determine lmd in GRL. Gradually change lmd from 0 to 1. (Depend on self.update_count)
    # ---------------------------------------------------------
    def calc_lambda(self):
        height = self.upper - self.lower
        progress = float(self.update_count) / float(self.max_iter)
        return 2.0 * height / (1.0 + math.exp(-self.alpha * progress)) - height + self.lower

    # ---------------------------------------------------------
    # forward process (Behave as identity function)
    # ---------------------------------------------------------
    def forward_cpu(self, x):
        return utils.force_array(x[0]),

    def forward_gpu(self, x):
        return cuda.cupy.array(x[0]),

    # ---------------------------------------------------------
    # backward process  (Multiple negative lmd)
    # ---------------------------------------------------------
    def backward_cpu(self, x, gy):
        self.update_count += 1
        self.calc_lambda()
        return utils.force_array(-self.current_lmd * gy[0]),

    def backward_gpu(self, x, gy):
        self.update_count += 1
        self.calc_lambda()
        return cuda.cupy.array(-self.current_lmd * gy[0]),


def flip_grad(x, iter_count, upper=1.0):
    return GRL(iter_count, upper)(x)
