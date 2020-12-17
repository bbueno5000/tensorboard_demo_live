"""
TODO: docstring
"""
import os 
import sys
import tensorflow
import urllib

class Tensorboard:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        self.logdir = '/tmp/mnist_tutorial/'
        self.github_url = \
            'https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'
        mnist = tensorflow.contrib.learn.datasets.mnist.read_data_sets(
            train_dir=self.logdir + 'data', one_hot=True)
        urllib.request.urlretrieve(
            self.github_url + 'labels_1024.tsv', self.logdir + 'labels_1024.tsv')
        urllib.request.urlretrieve(
            self.github_url + 'sprite_1024.png', self.logdir + 'sprite_1024.png')

    def __call__(self):
        """
        TODO: docstring
        """
        for learning_rate in [1E-4]:
            for use_two_fc in [True]:
                for use_two_conv in [True]:
                    hparam = self.make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                    print('Starting run for %s' % hparam)
                    self.mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)

    def conv_layer(self, input, size_in, size_out, name="conv"):
        """
        TODO: docstring
        """
        with tensorflow.name_scope(name):
            w = tensorflow.Variable(tensorflow.truncated_normal(
                [5, 5, size_in, size_out], stddev=0.1), name="W")
            b = tensorflow.Variable(tensorflow.constant(0.1, shape=[size_out]), name="B")
            conv = tensorflow.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            act = tensorflow.nn.relu(conv + b)
            tensorflow.summary.histogram("weights", w)
            tensorflow.summary.histogram("biases", b)
            tensorflow.summary.histogram("activations", act)
            return tensorflow.nn.max_pool(
                act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def fc_layer(self, input, size_in, size_out, name="fc"):
        """
        Dense (fully connected) layers perform classification on the features
        extracted by the convolutional layers and downsampled by the pooling layers.
        In a dense layer, every node in the layer is connected
        to every node in the preceding layer.
        """
        with tensorflow.name_scope(name):
            w = tensorflow.Variable(tensorflow.truncated_normal(
                [size_in, size_out], stddev=0.1), name="W")
            b = tensorflow.Variable(tensorflow.constant(0.1, shape=[size_out]), name="B")
            act = tensorflow.nn.relu(tensorflow.matmul(input, w) + b)
            tensorflow.summary.histogram("weights", w)
            tensorflow.summary.histogram("biases", b)
            tensorflow.summary.histogram("activations", act)
            return act

    def make_hparam_string(self, learning_rate, use_two_fc, use_two_conv):
        """
        TODO: docstring
        """
        conv_param = "conv=2" if use_two_conv else "conv=1"
        fc_param = "fc=2" if use_two_fc else "fc=1"
        return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

    def mnist_model(self, learning_rate, use_two_conv, use_two_fc, hparam):
        """
        Build our model.
        """
        tensorflow.reset_default_graph()
        sess = tensorflow.Session()
        x = tensorflow.placeholder(tensorflow.float32, shape=[None, 784], name="x")
        x_image = tensorflow.reshape(x, [-1, 28, 28, 1])
        tensorflow.summary.image('input', x_image, 3)
        y = tensorflow.placeholder(tensorflow.float32, shape=[None, 10], name="labels")
        if use_two_conv:
            conv1 = self.conv_layer(x_image, 1, 32, "conv1")
            conv_out = self.conv_layer(conv1, 32, 64, "conv2")
        else:
            conv1 = self.conv_layer(x_image, 1, 64, "conv")
            conv_out = tensorflow.nn.max_pool(
                conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        flattened = tensorflow.reshape(conv_out, [-1, 7 * 7 * 64])
        if use_two_fc:
            fc1 = self.fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
            embedding_input = fc1
            embedding_size = 1024
            logits = self.fc_layer(fc1, 1024, 10, "fc2")
        else:
            embedding_input = flattened
            embedding_size = 7 * 7 * 64
            logits = self.fc_layer(flattened, 7*7*64, 10, "fc")
        with tensorflow.name_scope("xent"):
            xent = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
            tensorflow.summary.scalar("xent", xent)
        with tensorflow.name_scope("train"):
           train_step = tensorflow.train.AdamOptimizer(learning_rate).minimize(xent)
        with tensorflow.name_scope("accuracy"):
            correct_prediction = tensorflow.equal(
                tensorflow.argmax(logits, 1), tensorflow.argmax(y, 1))
            accuracy = tensorflow.reduce_mean(
                tensorflow.cast(correct_prediction, tensorflow.float32))
            tensorflow.summary.scalar("accuracy", accuracy)
        summ = tensorflow.summary.merge_all()
        embedding = tensorflow.Variable(tensorflow.zeros(
            [1024, embedding_size]), name="test_embedding")
        assignment = embedding.assign(embedding_input)
        saver = tensorflow.train.Saver()
        sess.run(tensorflow.global_variables_initializer())
        writer = tensorflow.summary.FileWriter(self.logdir + hparam)
        writer.add_graph(sess.graph)
        config = tensorflow.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.sprite.image_path = self.logdir + 'sprite_1024.png'
        embedding_config.metadata_path = self.logdir + 'labels_1024.tsv'
        embedding_config.sprite.single_image_dim.extend([28, 28])
        tensorflow.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
        for i in range(2001):
            batch = self.mnist.train.next_batch(100)
            if i % 5 == 0:
                [train_accuracy, s] = sess.run(
                    [accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
                writer.add_summary(s, i)
            if i % 500 == 0:
                sess.run(assignment, feed_dict={
                    x: self.mnist.test.images[:1024], y: self.mnist.test.labels[:1024]})
                saver.save(sess, os.path.join(self.logdir, "model.ckpt"), i)
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def main(argv):
    """
    TODO: docstring
    """
    tensorboard = Tensorboard()
    tensorboard()

if __name__ == '__main__':
    main(sys.arg)
