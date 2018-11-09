import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

class MyDenseLyer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLyer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[input_shape[-1].value,
                                        self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


layer = MyDenseLyer(10)
print(layer(tf.zeros([10, 5])))
print(layer.variables)

