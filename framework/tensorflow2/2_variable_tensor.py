"""
介绍tensorflow中的变量跟张量
"""
import tensorflow as tf
import numpy as np


def variable():
    # 创建
    my_tensor = tf.constant([1.0, 2.0])
    my_variable = tf.Variable(my_tensor)     # 数值型
    bool_variable = tf.Variable([False, False, False, True])    # bool型
    complex_variable = tf.Variable([5 + 4j, 6 + 1j])    # 复数型
    print(my_variable)    # <tf.Variable 'Variable:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
    print(bool_variable)    # <tf.Variable 'Variable:0' shape=(4,) dtype=bool, numpy=array([False, False, False,  True])>
    print(complex_variable)    # <tf.Variable 'Variable:0' shape=(2,) dtype=complex128, numpy=array([5.+4.j, 6.+1.j])>


    # 再赋值
    my_variable.assign([2.0, 3.0])
    print(my_variable)    # <tf.Variable 'Variable:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>


def tensor():
    # 创建
    my_tensor1 = tf.constant([1.2, 4.5])
    my_numpy = my_tensor1.numpy()
    my_tensor2 = tf.convert_to_tensor(my_numpy)

    print(my_tensor1)    # tf.Tensor([1.2 4.5], shape=(2,), dtype=float32)
    print(my_numpy)    # [1.2 4.5]
    print(my_tensor2)    # tf.Tensor([1.2 4.5], shape=(2,), dtype=float32)

    # 形状
    rank_4_tensor = tf.zeros([3, 2, 4, 5])

    print("Type of every element:", rank_4_tensor.dtype)    # Type of every element: <dtype: 'float32'>
    print("Number of dimensions:", rank_4_tensor.ndim)    # Number of dimensions: 4
    print("Shape of tensor:", rank_4_tensor.shape)    # Shape of tensor: (3, 2, 4, 5)
    print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])    # Elements along axis 0 of tensor: 3
    print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])    # Elements along the last axis of tensor: 5
    print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())    # Total number of elements (3*2*4*5):  120

    # 改变形状
    new_shape_tensor = tf.reshape(rank_4_tensor, (3, 2, -1))
    print(new_shape_tensor.shape)    # (3, 2, 20)
    new_shape_tensor = tf.transpose(rank_4_tensor, [0, 1, 3, 2])
    print(new_shape_tensor.shape)    # (3, 2, 5, 4)

    # 索引
    rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]])

    print("Second row:", rank_2_tensor[1, :].numpy())    # Second row: [3 4]
    print("Second column:", rank_2_tensor[:, 1].numpy())    # Second column: [2 4 6]
    print("Last row:", rank_2_tensor[-1, :].numpy())    # Last row: [5 6]
    print("First item in last column:", rank_2_tensor[0, -1].numpy())    # First item in last column: 2
    print("Skip the first row:")
    print(rank_2_tensor[1:, :].numpy(), "\n")    # [[3 4] [5 6]]


def other_tensor():
    # 其他张量

    # 非矩阵张量
    ragged_list = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7, 8],
        [9]
    ]
    ragged_tensor = tf.ragged.constant(ragged_list)
    print(ragged_tensor)    # <tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
    print(ragged_tensor.shape)    # (4, None)

    # 稀疏张量
    sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                           values=[1, 2],
                                           dense_shape=[3, 4])
    print(sparse_tensor)
    # SparseTensor(indices=tf.Tensor( [[0 0] [1 2]], shape = (2, 2), dtype = int64), values = tf.Tensor([1 2], shape=(2,),dtype=int32), dense_shape = tf.Tensor([3 4],shape=(2,),dtype=int64))

    # 转为稠密矩阵
    print(tf.sparse.to_dense(sparse_tensor))
    # tf.Tensor([[1 0 0 0] [0 0 2 0] [0 0 0 0]], shape = (3, 4), dtype = int32)


def gradients():
    # GradientTape的参数与方法:
    # persistent: 控制tape的gradient()
    # 方法是否可以多次被调用，默认是False
    # watch_accessed_variables: 控制tape是否自动记录可训练的变量，默认是True
    # watch(): 手动添加被tape记录的变量（tensor）
    # watched_variables(): 按照构造顺序返回tape中被记录的变量
    # gradient(): 计算被tape记录的操作的梯度

    # 自动微分，梯度计算
    w = tf.Variable(tf.random.normal((3, 2)), name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]

    # 记录前向计算过程，保留在tape上
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w) + b
        loss = tf.reduce_mean(y ** 2)

    # 计算梯度
    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print(w.shape)    # (3, 2)
    print(dl_dw.shape)    # (3, 2)

    # 模型中的梯度计算
    layer = tf.keras.layers.Dense(2, activation='relu')
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(y ** 2)

    # 计算层（模型）中的所有变量的梯度
    grad = tape.gradient(loss, layer.trainable_variables)
    for var, g in zip(layer.trainable_variables, grad):
        # dense/kernel:0, shape: (3, 2)
        # dense/bias:0, shape: (2,)
        print(f'{var.name}, shape: {g.shape}')

