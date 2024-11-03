from core import Function
import numpy as np


class Square(Function):
    def forward(self, x):
        """
        :param x: 输入的数据，不是Variable类对象
        :return: 输入的数据，不是Variable类对象
        """
        y = x ** 2
        return y

    def backward(self, gy):
        """
        :param gy: 是这个函数对象产生的子节点x，对最终结果loss的导数，即d(loss)/dy，需要此参数完成链式法则
        :return:返回导数的Varible实例
        """
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


def square(x):
    """
    把计算函数对象创建和计算函数结果两个步骤合并为一个函数
    """
    f = Square()
    return f(x)

class Exp(Function):
    def forward(self, x):
        """
        输入输出都是数字，不是对象
        """
        y = np.exp(x)
        return y

    def backward(self, gy):

        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    """
    把计算函数对象创建和计算函数结果两个步骤合并为一个函数
    """
    f = Exp()
    return f(x)

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

