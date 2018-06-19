import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784 # 输入层节点数
OUTPUT_NODE = 10 # 输出层节点数 识别 0-9 所以上个月抽根烟

LAYER1_NODE = 500 # 隐藏层点数
BATCH_SIZE = 100 # 一个batch 中的训练个数

LEARNING_RATE_BASE = 0.8 # 学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 正则化项损失函数的系数
TRAINING_STEPS = 30000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    # 当没有提供滑动平均类时,是直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果,这里使用了Relu激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)

        # 计算输出层的前向传播结果,不用加relu 激活函数 , 因为计算损失函数会计算softmax函数
        return tf.matmul(layer1,weights2) + biases2
    else:
        # 先计算滑动平均值,再计算结果
        layer1_avg_weights1 = avg_class.average(weights1)
        layer1_avg_biases1 = avg_class.average(biases1)

        layer1 = tf.nn.relu(tf.matmul(input_tensor,layer1_avg_weights1) + layer1_avg_biases1)
        output1_avg_weights2 = avg_class.average(weights2)
        output1_avg_biases2 = avg_class.average(biases2)
        return tf.matmul(layer1,output1_avg_weights2) + output1_avg_biases2

def train(mnist):
    x = tf.placeholder(tf.float32, [None,INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name= 'y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))

    # 计算当前参数下的神经网络前向传播结果 , 先不用滑动平均类

    y = inference(x,None,weights1,biases1,weights2,biases2)

    # 定义储存训练轮数的变量 , 不可训练的变量
    global_step = tf.Variable(0,trainable= False)

    # 给定平均衰减率和训练轮数的变量,初始化
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均 , trainable , tf.trainable_variables 返回 GraphKeys.TRAINABLE_VARIABLES
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均下的神经网络前向传播结果
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    # 交叉熵作为预测值和真实值之间的损失函数

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

    # 计算当前batch样例的交叉熵(损失函数)平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 使用梯度下降优化损失函数,包括了交叉熵和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)


    # 每一次都要通过反向传播参数和参数滑动平均值 , 下面两种机制等价
    # with tf.control_dependencies([train_step,variable_averages_op]):
    #     train_op = tf.no_op(name='train')
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    # train_op = tf.group(train_step,variables_averages_op)


    #检查结果,将训练结果 averge_y 跟样本结果 y_ 做对比
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    # 先将布尔数值转换为实数,然后计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        # 训练啦
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 输出测试结果
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "" using average model is % g " % (i,validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d traning step(s) , test accuracy using average" "model is %g" % (TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()