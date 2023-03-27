import numpy as np
import tensorflow as tf
from keras.layers import Layer
from abc import ABC, abstractmethod


class MessagePassing(Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.input_dim = None
        self.edge_features_dim = None

    def build(self, input_shape):
        super().build(input_shape)
        self.input_dim = input_shape[0][1]
        self.edge_features_dim = input_shape[2][1]
        self.input_features_dim = 2 * self.input_dim + self.edge_features_dim

    def edge_function(self, features):
        return features

    def call(self, inputs):
        hidden_states_in, edges, edge_features = inputs
        features = tf.gather(hidden_states_in, edges)
        features = tf.reshape(features, [-1, 2 * self.input_dim])
        features = tf.concat([features, edge_features], 1)

        return self.edge_function(features)

    def get_config(self):
        config = super().get_config()
        config['output_dim'] = self.output_dim
        return config


class MessagePassingMLP2(MessagePassing):
    def __init__(self, output_dim, hidden_units=None, **kwargs):
        super().__init__(output_dim, **kwargs)
        if hidden_units is not None:
            self.hidden_units = hidden_units
        else:
            self.hidden_units = output_dim

    def build(self, input_shape):
        super().build(input_shape)
        self.w1 = self.add_weight(
            shape=(self.input_features_dim, self.hidden_units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(self.hidden_units, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(1, self.hidden_units)
        )

    def edge_function(self, features):
        out = tf.einsum('ij,jk->ik', features, self.w1)
        out += self.b1
        out = tf.nn.gelu(out)
        out = tf.einsum('ij,jk->ik', out, self.w2)
        return out


class HiddenStateUpdate(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = None
        self.message_dim = None
        self.input_feature_dim = None

    def build(self, input_shape):
        super().build(input_shape)

        self.input_dim = input_shape[0][1]
        self.message_dim = input_shape[2][1]
        self.input_feature_dim = self.input_dim + self.message_dim

    def aggregate_messages(self, messages, edges, num_points):
        message_out = tf.math.unsorted_segment_sum(messages, edges[:, 0], num_points)
        return message_out

    def update_hidden_states(self, hidden_states_in, message_aggr):
        return tf.concat([hidden_states_in, message_aggr], -1)

    def call(self, inputs):
        hidden_states_in, edges, messages = inputs
        num_points = tf.shape(hidden_states_in)[0]

        message_aggr = self.aggregate_messages(messages, edges, num_points)

        hidden_states_out = self.update_hidden_states(hidden_states_in, message_aggr)

        return hidden_states_out

class HiddenStateUpdateMLP2(HiddenStateUpdate):
    def __init__(self, output_dim, hidden_units=None, **kwargs):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        if hidden_units is not None:
            self.hidden_units = hidden_units
        else:
            self.hidden_units = output_dim


    def build(self, input_shape):
        super().build(input_shape)
        self.w1 = self.add_weight(
            shape=(self.input_features_dim, self.hidden_units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(self.hidden_units, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(1, self.hidden_units)
        )


    def aggregate_messages(self, messages, edges, num_points):
        message_out = tf.math.unsorted_segment_sum(messages, edges[:, 0], num_points)
        return message_out

    def update_hidden_states(self, hidden_states_in, message_aggr):
        features = tf.concat([hidden_states_in, message_aggr], -1)
        out = tf.einsum('ij,jk->ik', features, self.w1)
        out += self.b1
        out = tf.nn.gelu(out)
        out = tf.einsum('ij,jk->ik', out, self.w2)
        return out


if __name__ == '__main__':
    message_passing = MessagePassingMLP2(2)
    hidden_states_in = np.random.rand(5, 3)
    edges = np.array([[0, 1],
                      [1, 2],
                      [1, 3],
                      [2, 4],
                      [4, 3],
                      [3, 0]], dtype=np.int32)
    edge_features = np.random.rand(edges.shape[0], 1)
    messages = message_passing([hidden_states_in, edges, edge_features])
    print(messages.shape)
