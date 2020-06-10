import tensorflow as tf
from tensorflow.keras import layers,datasets
batch_size = 128
lr =0.01
epoches = 10
sum = 0
(x,y) ,(x_test,y_test) = datasets.mnist.load_data()

train_data = tf.data.Dataset.from_tensor_slices((x,y))
test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))

train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)
def process(x,y):
    x = tf.cast(x,dtype=tf.float32)/255

    y = tf.cast(y,dtype = tf.int64)
    # y = tf.one_hot(y,depth=10)
    return x,y

train_data = train_data.map(process)
test_data = test_data.map(process)
criteon = tf.losses.CategoricalCrossentropy(from_logits=True)
optimer = tf.optimizers.SGD(lr)
tf.optimizers.Adam()



# class LeNet(tf.keras.Model):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.con1 = layers.Conv2D(6,3,1)
#         self.cn2 = layers.Conv2D(16,3,1)


net = tf.keras.Sequential([layers.Conv2D(6,3,1),
                           layers.MaxPool2D(2,2),
                            layers.ReLU(),
                            layers.Conv2D(6,3,1),
                            layers.MaxPool2D(2,1),
                           layers.ReLU(),
                           layers.Flatten(),
                            layers.Dense(120,activation = 'relu'),
                           layers.Dense(84,activation='relu'),
                           layers.Dense(10)])

net.build((4,28,28,1))
net.summary()

def train(data,epoch):
    for i in range(epoch):
        for x, y in data:
            x = tf.expand_dims(x, axis=3)
            with tf.GradientTape() as tape:
                out = net(x)
                y = tf.one_hot(y,depth=10)
                loss = criteon(y,out)
            grads = tape.gradient(loss,net.trainable_variables)
            optimer.apply_gradients(zip(grads,net.trainable_variables))
        predict(test_data,net)

def predict(db_test,network):
    correct, total = 0, 0
    for x, y in db_test:
        # 遍历所有训练集样本
        # 插入通道维度，=>[b,28,28,1]
        x = tf.expand_dims(x, axis=3)  # 前向计算，获得 10 类别的预测分布，[b, 784] => [b, 10]
        out = network(x)
        # 真实的流程时先经过 softmax，再 argmax
        # 但是由于 softmax 不改变元素的大小相对关系，故省去
        pred = tf.argmax(out, axis=-1)
        y = tf.cast(y, tf.int64)
        # 统计预测正确数量
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
        # 统计预测样本总数
        total += x.shape[0]  # 计算准确率
    print('acc:',correct/total)

train(train_data,epoches)