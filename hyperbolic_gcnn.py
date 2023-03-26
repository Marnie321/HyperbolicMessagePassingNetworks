import numpy as np
import tensorflow as tf
from keras.layers import Layer

EPS = 1e-6


@tf.function
def minkowski_inner_product(x, y, dim=2, axis=-1, keepdims=False):
    w = tf.concat([tf.constant([-1], dtype=x.dtype), tf.ones(dim, dtype=x.dtype)], 0)
    shape = [1] * len(x.shape)
    shape[axis] = -1
    w = tf.reshape(w, shape)
    out = tf.reduce_sum(y * x * w, axis, keepdims=keepdims)
    return out


@tf.function
def minkowski_norm(x, dim=2, axis=-1, keepdims=False):
    """
    Computes the Minkowski norm by computing the Minkowski inner product of a point by itself.
    Disclaimer: the Minkowski inner product is not positive definite in the whole domain. It is
                positive definite in the tangent spaces of the hyperboloid though.
    """

    return tf.sqrt(minkowski_inner_product(x, x, dim, axis, keepdims) + EPS)


@tf.function
def hyperboloid_distance(x, y, k, dim=2, axis=-1, keepdims=False):
    return tf.math.acosh(-minkowski_inner_product(x, y, dim, axis, keepdims=keepdims) / k)


@tf.function
def exponential(x, v, k, dim=2, axis=-1):
    norm = minkowski_norm(v, dim, axis, keepdims=True)
    sqrtk = tf.sqrt(k)
    T = norm / sqrtk
    out = tf.math.cosh(T) * x + sqrtk * tf.math.sinh(T) * (v / norm)
    return out


@tf.function
def log(x, y, k, dim=2, axis=-1):
    t = minkowski_inner_product(x, y, dim, axis, keepdims=True)
    out = y + x * t / k
    norm = minkowski_norm(out, dim, axis, keepdims=True)
    out = out / norm
    return out * hyperboloid_distance(x, y, k, dim, axis, keepdims=True)


class HyperboloidAggregation(Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)

        self.dim = None
        self.k = k

    def build(self, input_shape):
        self.dim = input_shape[0][1]
        super().build(input_shape)

    def vertex_function(self, messages):
        return tf.reduce_mean(messages, -1, keepdims=True)

    def call(self, inputs):
        vertices, edges, messages = inputs
        num_points = tf.shape(vertices)[0]
        weights = self.vertex_function(messages)
        vertex_pairs = tf.gather(vertices, edges)
        vectors = log(vertex_pairs[:, 0, :], vertex_pairs[:, 1, :], self.k)
        vectors = weights * vectors
        tangent_aggregations = tf.math.unsorted_segment_mean(vectors, edges[:, 0], num_points)
        return exponential(vertices, tangent_aggregations, self.k)

    def get_config(self):
        config = super().get_config()
        config['k'] = self.k


class HyperboloidAggregationMLP2(HyperboloidAggregation):
    def __init__(self, k, hidden_units, **kwargs):
        super().__init__(k, **kwargs)
        self.hidden_units = hidden_units

    def build(self, input_shape):
        super().build(input_shape)
        self.w1 = self.add_weight(
            shape=(self.dim, self.hidden_units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(self.hidden_units,),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(1, self.hidden_units)
        )

    def vertex_function(self, features):
        out = tf.einsum('ij,jk->ik', features, self.w1)
        out += self.b1
        out = tf.nn.gelu(out)
        out = tf.einsum('ij,j->i', out, self.w2)
        return tf.expand_dims(out, 1)


if __name__ == '__main__':
    hyper_aggr = HyperboloidAggregationMLP2(1., 3)

    vertices_in = 3 * np.random.rand(5, 3).astype(np.float32)
    vertices_in[:, 0] = 0.
    origin = np.array([(1., 0, 0)], np.float32)
    vertices_in = exponential(origin, vertices_in, 1.)
    print(vertices_in)
    print(minkowski_inner_product(vertices_in, vertices_in))

    edges = np.array([[0, 1],
                      [1, 2],
                      [1, 3],
                      [2, 4],
                      [4, 3],
                      [3, 0]], dtype=np.int32)
    messages = np.random.rand(edges.shape[0], 3)

    vertices_out = hyper_aggr([vertices_in, edges, messages])
    print(vertices_out)
    print(vertices_out.shape)
    print(minkowski_inner_product(vertices_out, vertices_out))
