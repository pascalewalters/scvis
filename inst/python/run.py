import numpy as np
import os
import yaml
import pandas as pd
import tensorflow as tf
from datetime import datetime
from collections import namedtuple

try:
    import cPickle as pickle
except ImportError:
    import pickle


def train(args):

    x, y, architecture, hyperparameter, train_data, model, normalizer, out_dir, name = \
        _init_model(args, 'train')
    iter_per_epoch = round(x.shape[0] / hyperparameter['batch_size'])

    max_iter = int(iter_per_epoch * hyperparameter['max_epoch'])

    if max_iter < 3000:
        max_iter = 3000
    elif max_iter > 30000:
        max_iter = np.max([30000, iter_per_epoch * 2])

    name += '_iter_' + str(max_iter)
    res = model.train(data=train_data,
                      batch_size=hyperparameter['batch_size'],
                      max_iter=max_iter,
                      pretrained_model=args['pretrained_model_file'])
    model.set_normalizer(normalizer)

    # Save the trained model
    out_dir = args['out_dir']
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model_dir = os.path.join(out_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_name = name + ".ckpt"
    model_name = os.path.join(model_dir, model_name)
    model.save_sess(model_name)

    # The objective function trace plot
    elbo = res['elbo']
    tsne_cost = res['tsne_cost']

    iteration = len(elbo)
    avg_elbo = elbo - tsne_cost
    for i in range(iteration)[1:]:
        avg_elbo[i] = (elbo[i] - tsne_cost[i]) / i + \
                   avg_elbo[i-1] * (i-1) / i

    obj_file = name + '_obj.tsv'
    obj_file = os.path.join(out_dir, obj_file)
    res = pd.DataFrame(np.column_stack((elbo, tsne_cost)),
                       columns=['elbo', 'tsne_cost'])
    res.to_csv(obj_file, sep='\t', index=True, header=True)

    # Save the mapping results
    _save_result(x, y, model, out_dir, name)

    return()


def map(args):
    x, y, architecture, hyperparameter, train_data, model, _, out_dir, name = \
        _init_model(args, 'map')

    name = "_".join([name, "map"])
    _save_result(x, y, model, out_dir, name)

    return()


def _init_model(args, mode):
    x = pd.read_csv(args['data_matrix_file'], sep='\t').values

    config = {}
    try:
        config_file_yaml = open(args['config_file'], 'r')
        config = yaml.load(config_file_yaml)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print('Error in the configuration file: {}'.format(exc))

    architecture = config['architecture']
    architecture.update({'input_dimension': x.shape[1]})

    hyperparameter = config['hyperparameter']
    if hyperparameter['batch_size'] > x.shape[0]:
        hyperparameter.update({'batch_size': x.shape[0]})

    model = SCVIS(architecture, hyperparameter)
    normalizer = 1.0
    if args['pretrained_model_file'] is not None:
        model.load_sess(args['pretrained_model_file'])
        normalizer = model.get_normalizer()

    if mode == 'train':
        if args['normalize'] is not None:
            normalizer = float(args['normalize'])
        else:
            normalizer = np.max(np.abs(x))
    else:
        if args['normalize'] is not None:
            normalizer = float(args['normalize'])

    x /= normalizer

    y = None
    if args['data_label_file'] is not None:
        label = pd.read_csv(args['data_label_file'], sep='\t').values
        label = pd.Categorical(label[:, 0])
        y = label.codes

    # fixed random seed
    np.random.seed(0)
    train_data = DataSet(x, y)

    out_dir = args['out_dir']
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    name = '_'.join(['perplexity', str(hyperparameter['perplexity']),
                     'regularizer', str(hyperparameter['regularizer_l2']),
                     'batch_size', str(hyperparameter['batch_size']),
                     'learning_rate', str(hyperparameter['optimization']['learning_rate']),
                     'latent_dimension', str(architecture['latent_dimension']),
                     'activation', str(architecture['activation']),
                     'seed', str(hyperparameter['seed'])])

    return x, y, architecture, hyperparameter, train_data, model, normalizer, out_dir, name


def _save_result(x, y, model, out_dir, name):
    z_mu, _ = model.encode(x)
    z_mu = pd.DataFrame(z_mu, columns=['z_coordinate_'+str(i) for i in range(z_mu.shape[1])])
    map_name = name + '.tsv'
    map_name = os.path.join(out_dir, map_name)
    z_mu.to_csv(map_name, sep='\t', index=True, header=True)

    log_likelihood = model.get_log_likelihood(x)
    log_likelihood = pd.DataFrame(log_likelihood, columns=['log_likelihood'])
    map_name = name + '_log_likelihood' + '.tsv'
    map_name = os.path.join(out_dir, map_name)
    log_likelihood.to_csv(map_name, sep='\t', index=True, header=True)


# =============================================================================
class SCVIS(object):
    def __init__(self, architecture, hyperparameter):
        self.eps = 1e-20

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.normalizer = tf.Variable(1.0, name='normalizer', trainable=False)

        self.architecture, self.hyperparameter = architecture, hyperparameter
        self.regularizer_l2 = self.hyperparameter['regularizer_l2']
        self.n = self.hyperparameter['batch_size']
        self.perplexity = self.hyperparameter['perplexity']

        tf.set_random_seed(self.hyperparameter['seed'])

        # Place_holders
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.x = tf.placeholder(tf.float32, shape=[None, self.architecture['input_dimension']])
        self.z = tf.placeholder(tf.float32, shape=[None, self.architecture['latent_dimension']])

        self.p = tf.placeholder(tf.float32, shape=[None, None])
        self.iter = tf.placeholder(dtype=tf.float32)

        self.vae = GaussianVAE(self.x,
                               self.batch_size,
                               self.architecture['inference']['layer_size'],
                               self.architecture['latent_dimension'],
                               decoder_layer_size=self.architecture['model']['layer_size'])

        self.encoder_parameter = self.vae.encoder_parameter
        self.latent = dict()
        self.latent['mu'] = self.encoder_parameter.mu
        self.latent['sigma_square'] = self.encoder_parameter.sigma_square
        self.latent['sigma'] = tf.sqrt(self.latent['sigma_square'])

        self.decoder_parameter = self.vae.decoder_parameter
        self.dof = tf.Variable(tf.constant(1.0, shape=[self.architecture['input_dimension']]),
                               trainable=True, name='dof')
        self.dof = tf.clip_by_value(self.dof, 0.1, 10, name='dof')

        with tf.name_scope('ELBO'):
            self.weight = tf.clip_by_value(tf.reduce_sum(self.p, 0), 0.01, 2.0)

            self.log_likelihood = tf.reduce_mean(tf.multiply(
                log_likelihood_student(self.x,
                                       self.decoder_parameter.mu,
                                       self.decoder_parameter.sigma_square,
                                       self.dof),
                self.weight), name="log_likelihood")

            self.kl_divergence = \
                tf.reduce_mean(0.5 * tf.reduce_sum(self.latent['mu'] ** 2 +
                                                   self.latent['sigma_square'] -
                                                   tf.log(self.latent['sigma_square']) - 1,
                                                   reduction_indices=1))
            self.kl_divergence *= tf.maximum(0.1, self.architecture['input_dimension']/self.iter)
            self.elbo = self.log_likelihood - self.kl_divergence

        self.z_batch = self.vae.z

        with tf.name_scope('tsne'):
            self.kl_pq = self.tsne_repel() * tf.minimum(self.iter, self.architecture['input_dimension'])

        with tf.name_scope('objective'):
            self.obj = self.kl_pq + self.regularizer() - self.elbo

        # Optimization
        with tf.name_scope('optimizer'):
            learning_rate = self.hyperparameter['optimization']['learning_rate']

            if self.hyperparameter['optimization']['method'].lower() == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif self.hyperparameter['optimization']['method'].lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate,
                                                        beta1=0.9,
                                                        beta2=0.999,
                                                        epsilon=0.001)

            gradient_clipped = self.clip_gradient()

            self.train_op = self.optimizer.apply_gradients(gradient_clipped, name='minimize_cost')

        self.saver = tf.train.Saver()

    def clip_gradient(self, clip_value=3.0, clip_norm=10.0):
        trainable_variable = self.sess.graph.get_collection('trainable_variables')
        grad_and_var = self.optimizer.compute_gradients(self.obj, trainable_variable)

        grad_and_var = [(grad, var) for grad, var in grad_and_var if grad is not None]
        grad, var = zip(*grad_and_var)
        grad, global_grad_norm = tf.clip_by_global_norm(grad, clip_norm=clip_norm)

        grad_clipped_and_var = [(tf.clip_by_value(grad[i], -clip_value*0.1, clip_value*0.1), var[i])
                                if 'encoder-sigma' in var[i].name
                                else (tf.clip_by_value(grad[i], -clip_value, clip_value), var[i])
                                for i in range(len(grad_and_var))]

        return grad_clipped_and_var

    def regularizer(self):
        penalty = [tf.nn.l2_loss(var) for var in
                   self.sess.graph.get_collection('trainable_variables')
                   if 'weight' in var.name]

        l2_regularizer = self.regularizer_l2 * tf.add_n(penalty)

        return l2_regularizer

    def tsne_repel(self):
        nu = tf.constant(self.architecture['latent_dimension'] - 1, dtype=tf.float32)

        sum_y = tf.reduce_sum(tf.square(self.z_batch), reduction_indices=1)
        num = -2.0 * tf.matmul(self.z_batch,
                               self.z_batch,
                               transpose_b=True) + tf.reshape(sum_y, [-1, 1]) + sum_y
        num = num / nu

        p = self.p + 0.1 / self.n
        p = p / tf.expand_dims(tf.reduce_sum(p, reduction_indices=1), 1)

        num = tf.pow(1.0 + num, -(nu + 1.0) / 2.0)
        attraction = tf.multiply(p, tf.log(num))
        attraction = -tf.reduce_sum(attraction)

        den = tf.reduce_sum(num, reduction_indices=1) - 1
        repellant = tf.reduce_sum(tf.log(den))

        return (repellant + attraction) / self.n

    def _train_batch(self, x, t):
        p = compute_transition_probability(x, perplexity=self.perplexity)

        feed_dict = {self.x: x,
                     self.p: p,
                     self.batch_size: x.shape[0],
                     self.iter: t}

        _, elbo, tsne_cost = self.sess.run([
            self.train_op,
            self.elbo,
            self.kl_pq],
            feed_dict=feed_dict)

        return elbo, tsne_cost

    def train(self, data, max_iter=1000, batch_size=None,
              pretrained_model=None, verbose=True, verbose_interval=50):

        max_iter = max_iter
        batch_size = batch_size or self.hyperparameter['batch_size']

        status = dict()
        status['elbo'] = np.zeros(max_iter)
        status['tsne_cost'] = np.zeros(max_iter)

        if pretrained_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.load_sess(pretrained_model)

        start = datetime.now()
        for iter_i in range(max_iter):
            x, y = data.next_batch(batch_size)

            status_batch = self._train_batch(x, iter_i+1)
            status['elbo'][iter_i] = status_batch[0]
            status['tsne_cost'][iter_i] = status_batch[1]

            if verbose and iter_i % verbose_interval == 0:
                print('Batch {}'.format(iter_i))
                print((
                    'elbo: {}\n'
                    'scaled_tsne_cost: {}\n').format(
                    status['elbo'][iter_i],
                    status['tsne_cost'][iter_i]))

        return status

    def encode(self, x):
        var = self.vae.encoder(prob=1.0)
        feed_dict = {self.x: x}

        return self.sess.run(var, feed_dict=feed_dict)

    def decode(self, z):
        var = self.vae.decoder(tf.cast(z, tf.float32))
        feed_dict = {self.z: z, self.batch_size: z.shape[0]}

        return self.sess.run(var, feed_dict=feed_dict)

    def encode_decode(self, x):
        var = [self.latent['mu'],
               self.latent['sigma_square'],
               self.decoder_parameter.mu,
               self.decoder_parameter.sigma_square]

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}

        return self.sess.run(var, feed_dict=feed_dict)

    def save_sess(self, model_name):
        self.saver.save(self.sess, model_name)

    def load_sess(self, model_name):
        self.saver.restore(self.sess, model_name)

    def get_log_likelihood(self, x, dof=None):

        dof = dof or self.dof
        log_likelihood = log_likelihood_student(
            self.x,
            self.decoder_parameter.mu,
            self.decoder_parameter.sigma_square,
            dof
        )
        num_samples = 5

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}
        log_likelihood_value = 0

        for i in range(num_samples):
            log_likelihood_value += self.sess.run(log_likelihood, feed_dict=feed_dict)

        log_likelihood_value /= np.float32(num_samples)

        return log_likelihood_value

    def get_elbo(self, x):
        log_likelihood = log_likelihood_student(
            self.x,
            self.decoder_parameter.mu,
            self.decoder_parameter.sigma_square,
            self.dof
        )
        kl_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(self.latent['mu'] ** 2 +
                                                           self.latent['sigma_square'] -
                                                           tf.log(self.latent['sigma_square']) - 1,
                                       reduction_indices=1))

        feed_dict = {self.x: x, self.batch_size: x.shape[0]}

        return self.sess.run(log_likelihood - kl_divergence, feed_dict=feed_dict)

    def set_normalizer(self, normalizer=1.0):
        normalizer_op = self.normalizer.assign(normalizer)
        self.sess.run(normalizer_op)

    def get_normalizer(self):
        return self.sess.run(self.normalizer)

MAX_VALUE = 1
LABEL = None

np.random.seed(0)


class DataSet(object):
    def __init__(self, x, y=LABEL, max_value=MAX_VALUE):
        if y is not None:
            assert x.shape[0] == y.shape[0], \
                ('x.shape: %s, y.shape: %s' % (x.shape, y.shape))

        self._num_data = x.shape[0]
        x = x.astype(np.float32)
        x /= max_value

        self._x = x
        self._y = y
        self._epoch = 0
        self._index_in_epoch = 0

        index = np.arange(self._num_data)
        np.random.shuffle(index)
        self._index = index

    @property
    def all_data(self):
        return self._x

    @property
    def label(self):
        return self._y

    @property
    def num_of_data(self):
        return self._num_data

    @property
    def completed_epoch(self):
        return self._epoch

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_data:
            assert batch_size <= self._num_data, \
                ('batch_size: %s, num_data: %s' % (batch_size, self._num_data))

            self._epoch += 1
            np.random.shuffle(self._index)

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        index = self._index[start:end]
        if self._y is not None:
            y = self._y[index]
        else:
            y = self._y

        return self._x[index], y


LAYER_SIZE = [128, 64, 32]
OUTPUT_DIM = 2
KEEP_PROB = 1.0
EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10

LocationScale = namedtuple('LocationScale', ['mu', 'sigma_square'])

# =============================================================================
def weight_xavier_relu(fan_in_out, name='weight'):
    stddev = tf.cast(tf.sqrt(2.0 / fan_in_out[0]), tf.float32)
    initial_w = tf.truncated_normal(shape=fan_in_out,
                                    mean=0.0, stddev=stddev)

    return tf.Variable(initial_w, trainable=True, name=name)

def bias_variable(fan_in_out, mean=0.1, name='bias'):
    initial = tf.constant(mean, shape=fan_in_out)
    return tf.Variable(initial, trainable=True, name=name)


def shape(tensor):
    return tensor.get_shape().as_list()


class MLP(object):
    def __init__(self, input_data, input_size, layer_size, output_dim,
                 activate_op=tf.nn.elu,
                 init_w_op=weight_xavier_relu,
                 init_b_op=bias_variable):
        self.input_data = input_data
        self.input_dim = shape(input_data)[1]
        self.input_size = input_size

        self.layer_size = layer_size
        self.output_dim = output_dim

        self.activate, self.init_w, self.init_b = \
            activate_op, init_w_op, init_b_op

        with tf.name_scope('encoder-net'):
            self.weights = [self.init_w([self.input_dim, layer_size[0]])]
            self.biases = [self.init_b([layer_size[0]])]

            self.hidden_layer_out = \
                tf.matmul(self.input_data, self.weights[-1]) + self.biases[-1]
            self.hidden_layer_out = self.activate(self.hidden_layer_out)

            for in_dim, out_dim in zip(layer_size, layer_size[1:]):
                self.weights.append(self.init_w([in_dim, out_dim]))
                self.biases.append(self.init_b([out_dim]))
                self.hidden_layer_out = self.activate(
                    tf.matmul(self.hidden_layer_out, self.weights[-1]) +
                    self.biases[-1])

class GaussianVAE(MLP):
    def __init__(self, input_data, input_size,
                 layer_size=LAYER_SIZE,
                 output_dim=OUTPUT_DIM,
                 decoder_layer_size=LAYER_SIZE[::-1]):
        super(self.__class__, self).__init__(input_data, input_size,
                                             layer_size, output_dim)

        self.num_encoder_layer = len(self.layer_size)

        with tf.name_scope('encoder-mu'):
            self.bias_mu = self.init_b([self.output_dim])
            self.weights_mu = self.init_w([self.layer_size[-1], self.output_dim])

        with tf.name_scope('encoder-sigma'):
            self.bias_sigma_square = self.init_b([self.output_dim])
            self.weights_sigma_square = self.init_w([self.layer_size[-1], self.output_dim])

        with tf.name_scope('encoder-parameter'):
            self.encoder_parameter = self.encoder()

        with tf.name_scope('sample'):
            self.ep = tf.random_normal(
                [self.input_size, self.output_dim],
                mean=0, stddev=1, name='epsilon_univariate_norm')

            self.z = tf.add(self.encoder_parameter.mu,
                            tf.sqrt(self.encoder_parameter.sigma_square) * self.ep,
                            name='latent_z')

        self.decoder_layer_size = decoder_layer_size
        self.num_decoder_layer = len(self.decoder_layer_size)

        with tf.name_scope('decoder'):
            self.weights.append(self.init_w([self.output_dim, self.decoder_layer_size[0]]))
            self.biases.append(self.init_b([self.decoder_layer_size[0]]))

            self.decoder_hidden_layer_out = self.activate(
                tf.matmul(self.z, self.weights[-1]) +
                self.biases[-1])

            for in_dim, out_dim in \
                    zip(self.decoder_layer_size, self.decoder_layer_size[1:]):
                self.weights.append(self.init_w([in_dim, out_dim]))
                self.biases.append(self.init_b([out_dim]))

                self.decoder_hidden_layer_out = self.activate(
                    tf.matmul(self.decoder_hidden_layer_out, self.weights[-1]) +
                    self.biases[-1])

            self.decoder_bias_mu = self.init_b([self.input_dim])
            self.decoder_weights_mu = \
                self.init_w([self.decoder_layer_size[-1],
                             self.input_dim])

            self.decoder_bias_sigma_square = self.init_b([self.input_dim])
            self.decoder_weights_sigma_square = \
                self.init_w([self.decoder_layer_size[-1],
                             self.input_dim])

            mu = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                  self.decoder_weights_mu),
                        self.decoder_bias_mu)
            sigma_square = tf.add(tf.matmul(self.decoder_hidden_layer_out,
                                            self.decoder_weights_sigma_square),
                                  self.decoder_bias_sigma_square)

            self.decoder_parameter = \
                LocationScale(mu, tf.clip_by_value(tf.nn.softplus(sigma_square),
                                                   EPS, MAX_SIGMA_SQUARE))

    def decoder(self, z):
        hidden_layer_out = self.activate(
            tf.matmul(z, self.weights[self.num_encoder_layer]) +
            self.biases[self.num_encoder_layer]
        )

        for layer in range(self.num_encoder_layer+1,
                           self.num_encoder_layer + self.num_decoder_layer):
            hidden_layer_out = self.activate(
                tf.matmul(hidden_layer_out, self.weights[layer]) +
                self.biases[layer])

        mu = tf.add(tf.matmul(hidden_layer_out, self.decoder_weights_mu),
                    self.decoder_bias_mu)
        sigma_square = tf.add(tf.matmul(hidden_layer_out,
                                        self.decoder_weights_sigma_square),
                              self.decoder_bias_sigma_square)

        return LocationScale(mu, tf.clip_by_value(tf.nn.softplus(sigma_square),
                                                  EPS, MAX_SIGMA_SQUARE))

    def encoder(self, prob=0.9):
        weights_mu = tf.nn.dropout(self.weights_mu, prob)
        mu = tf.add(tf.matmul(self.hidden_layer_out, weights_mu),
                    self.bias_mu)
        sigma_square = tf.add(tf.matmul(self.hidden_layer_out,
                                        self.weights_sigma_square),
                              self.bias_sigma_square)

        return LocationScale(mu,
                             tf.clip_by_value(tf.nn.softplus(sigma_square),
                                              EPS, MAX_SIGMA_SQUARE))

def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = tf.sqrt(sigma_square)

    dist = tf.contrib.distributions.StudentT(df=df,
                                             loc=mu,
                                             scale=sigma)
    return tf.reduce_sum(dist.log_prob(x), reduction_indices=1)


def compute_transition_probability(x, perplexity=30.0,
                                   tol=1e-4, max_iter=50, verbose=False):
    # x should be properly scaled so the distances are not either too small or too large

    if verbose:
        print('tSNE: searching for sigma ...')

    (n, d) = x.shape
    sum_x = np.sum(np.square(x), 1)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))

    # Parameterized by precision
    beta = np.ones((n, 1))
    entropy = np.log(perplexity) / np.log(2)

    # Binary search for sigma_i
    idx = range(n)
    for i in range(n):
        idx_i = list(idx[:i]) + list(idx[i+1:n])

        beta_min = -np.inf
        beta_max = np.inf

        # Remove d_ii
        dist_i = dist[i, idx_i]
        h_i, p_i = compute_entropy(dist_i, beta[i])
        h_diff = h_i - entropy

        iter_i = 0
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta[i].copy()
                if np.isfinite(beta_max):
                    beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta[i] *= 2.0
            else:
                beta_max = beta[i].copy()
                if np.isfinite(beta_min):
                    beta[i] = (beta[i] + beta_min) / 2.0
                else:
                    beta[i] /= 2.0

            h_i, p_i = compute_entropy(dist_i, beta[i])
            h_diff = h_i - entropy

            iter_i += 1

        p[i, idx_i] = p_i

    if verbose:
        print('Min of sigma square: {}'.format(np.min(1 / beta)))
        print('Max of sigma square: {}'.format(np.max(1 / beta)))
        print('Mean of sigma square: {}'.format(np.mean(1 / beta)))

    return p

MAX_VAL = np.log(sys.float_info.max) / 2.0

np.random.seed(0)

def compute_entropy(dist=np.array([]), beta=1.0):
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p

    return h, p / sum_p

