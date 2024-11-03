import numpy as np
import weakref
import contextlib

'''
===============================================================


                    两个基类+是否启用反向传播的Config类


===============================================================
'''


class Config:
    # 默认值为True，只有在调用with结构中的no_grad()函数临时修改为False
    # 调用完成之后会自动恢复为True
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    # name是说Config类中的哪个属性名字
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        # 禁用反向传播的代码,yield表示with函数中代码块的代码
        yield
    finally:
        # 恢复Config值的代码
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Function:
    def __call__(self, *inputs):
        """
        :param input: 是一个Variable类的对象,不是数字
        :return: 是一个Variable类的对象,不是数字
        """
        # 统一输入数据类型为Variable变量，使得Function函数可以与常数、numpy数组等非Variable对象直接运算
        inputs = [as_variable(x) for x in inputs]

        # 进行前向传播
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        # 用于统一前向传播的结果节点数值数据类型为numpy类型
        if not isinstance(ys, tuple):
            ys = (ys,)                      # 统一输出的数值为元组类型
        outputs = [Variable(as_array(y)) for y in ys]

        # 是否启用反向传播
        if Config.enable_backprop:
            # 设置辈分变量，用于指导Func栈的出栈顺序
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)        # 这里不直接修改属性，用函数修改，方便以后维护
            self.inputs = inputs
            # self.outputs = outputs            # 避免使用强引用导致循环引用，优化内存占用
            self.outputs = [weakref.ref(output) for output in outputs]  # 由于需要对每一个对象是用weakref，所以不能直接用outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        # 注意，这里的xs是解包过后的元素，如果要使用，例如对于Add而言，x1, x2 = xs
        raise NotImplementedError()     # 此方法需要被继承才能使用

    def backward(self, gy):
        raise NotImplementedError()     # 此方法需要被继承才能使用


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            # 用于限定，输入数据只能是numpy类的数据，或者None
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None        # 输出结果y对该变量的导数，即dy/dx
        self.creator = None     # 得到这个变量的函数，即计算图中得到这个节点的Function节点的对象(例如a+b的+与c+d的+是不同的对象)
        self.generation = 0

    def set_creator(self, func):
        self.creator = func     # 设置节点的creator属性，在前向传播产生本节点的时候调用
        self.generation = func.generation + 1       # 设置辈分变量，使得反向传播的时候，理清楚先更新哪个结点的梯度

    def backward(self, retain_grad=False, create_graph=False):
        # retain_grad参数表示是否需要保存计算图中中间节点的梯度值，如果为false，则当这个节点没有用了之后，梯度会置空为None，用于优化内存
        # create_graph参数，当此参数为False时，在反向传播计算时，在“禁用反向传播模式”下进行反向传播计算。即只进行一次反向传播
        if self.grad is None:
            '''
            如果要创建反向传播计算图，我们需要用Variable实例代替ndarray实例进行计算，
            所以我们需要把梯度保存为Variable实例
            '''
            #self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()    # 用集合set，防止同一个func被多次添加到计算图的反向传播栈中

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)      # 按照辈分进行排序

        add_func(self.creator)

        while funcs:            # funcs栈不为空，则反向传播还没有结束
            f = funcs.pop()     # 获取当前循环要操作的函数
            gys = [output().grad for output in f.outputs]           # 为了消除循环引用，用output()

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):    # 让inputs和gxs两个列表一一对应地遍历
                    # 防止结点被重复使用的时候，梯度被覆盖
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)

            # 优化内存，去除中间节点（没有用了的节点）的梯度，由于存在funcs栈的通过辈分排序的关系
            # 能够保证运行到这行代码的时候f.outputs中的节点不再被需要
            if not retain_grad:
                for y in f.outputs:
                    # 由于采用了weakref的弱引用的引用方式消除循环引用，f.outputs中的数据不是直接引用
                    # y的属性不能直接用y.grad访问，必须通过y()访问，所以是y().grad。详情请参考步骤17中对于weakref的讲述
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    # 让变量更易用的代码 shape、ndim、size、dtype函数
    # @property使得shape函数能作为属性直接使用如x.shape而不是x.shape()
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.size

    def __len__(self):
        return len(self.data)

    # 便于print(节点)
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'

    # 提高Variable结点在运算的时候的优先级，使其高于numpy结点（这里不是很懂）,为什么一定要优先运算Variable呢？
    # 加减乘除不都是左结合吗？还有为什么设置了这个参数，优先级就变高了？
    __array_priority__ = 200


'''
===============================================================


                            一些辅助函数（用于类型转换与统一类型）


===============================================================
'''


# 将变量的类型调整为ndarray类型
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 将非Variable对象的ndarray和int float等数据类型转化为Variable对象
# 方便Function对象与int、ndarray等进行运算
# 这里的obj只能是Variable变量、ndarray和可以用于生成numpy数组类型的变量（例如int、float）
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


'''
===============================================================


            基本计算函数（四则运算）
        Mul,Div,Pow的反向传播返回值修改为Variable


===============================================================
'''


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0,x1):
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return x1 * gy, x0 * gy


def mul(x0, x1):
    x1 = as_array(x1)
    f = Mul()
    return f(x0, x1)


# 负数的取相反数运算（单操作数）
class Neg(Function):
    def forward(self, x):
        # xs是值，不是Variable对象
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    x = as_array(x)        # 这行书上没有，我自己加的（陈博勋）
    f = Neg()
    return f(x)


# 减法运算
class Sub(Function):
    def forward(self, x0, x1):
        # x0, x1是操作数的数值，不是Variable对象
        y = x0 - x1
        return y

    def backward(self, gy):
        # gy是导数的数值，不是Variable对象
        return gy, -gy


def sub(x0, x1):
    # 表示x0 - x1，注意先后顺序，需要严格定义使用__sub__与__rsub__的优先级
    x1 = as_array(x1)
    f = Sub()
    return f(x0, x1)


def rsub(x0, x1):
    # 如果直接认为__rsub__等于__sub__，得到的是正确答案的相反数
    # 因为减法不可交换
    x1 = as_array(x1)
    return Sub()(x1, x0)    # 这里调换了x0和x1的顺序


# 除法
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return 1 / x1 * gy, -x0 * gy / (x1 ** 2)


def div(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        c = self.c
        x = self.inputs[0]
        return c * (x ** (c - 1)) * gy


def pow(x, c):
    f = Pow(c)
    return f(x)


'''
===============================================================


                            运算符重载


===============================================================
'''


# python中函数也是对象，所以可以赋值
# 两个操作数的时候，优先按照是Variable对象选择__operator__还是__roperator__
# 如果左操作数为Variable对象，并且右操作数为其他，调用的是__operator__
# 相反，调用的是__roperator__
def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
