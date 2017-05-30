# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf
from tensorflow import logging


class BaseLoss(object):
    """Inherit from this class when implementing new losses."""

    def calculate_loss(self, unused_predictions, unused_labels, unused_labels_audio, **unused_params):
        """Calculates the average loss of the examples in a mini-batch.

         Args:
          unused_predictions: a 2-d tensor storing the prediction scores, in which
            each row represents a sample in the mini-batch and each column
            represents a class.
          unused_labels: a 2-d tensor storing the labels, which has the same shape
            as the unused_predictions. The labels must be in the range of 0 and 1.
          unused_params: loss specific parameters.

        Returns:
          A scalar loss tensor.
        """
        raise NotImplementedError()


class CosineLoss(BaseLoss):
    """Calculate the cosine distance loss between the predictions and labels.
    """

    def calculate_loss(self, embedding, labels, **unused_params):
        with tf.name_scope("loss_cosine"):
            embedding_audio = embedding[:, 0:128]
            embedding_frames = embedding[:, 128:2 * 128]
            # We do the normalization in the model
            # embedding_frames = tf.nn.l2_normalize(embedding_frames, 1)
            # embedding_audio = tf.nn.l2_normalize(embedding_audio, 1)
            return tf.losses.cosine_distance(embedding_audio, embedding_frames, dim=1)


class CosineAndCrossEntropyLoss(BaseLoss):
    """Calculate the cosine distance loss between the predictions and labels.
    """

    def calculate_loss(self, predictions, labels, labels_audio, embeddings=[], vocab_size=4716, reg_lambda=0.00,
                       margin=0.3,is_negative=[], **unused_params):
        with tf.name_scope("loss_cosine_entropy"):
            is_negative_float = tf.to_float(is_negative)

            # We do the normalization in the model
            # embedding_frames = tf.nn.l2_normalize(embedding_frames, 1)
            # embedding_audio = tf.nn.l2_normalize(embedding_audio, 1)
            embedding_audio = embeddings[:, 0:128]
            embedding_frames = embeddings[:, 128:2 * 128]

            predictions_audio = predictions[:, 0:vocab_size]
            predictions_frames = predictions[:, vocab_size:2 * vocab_size]

            # Cosine loss
            cosine_loss = tf.losses.cosine_distance(embedding_audio, embedding_frames, dim=1)

            # Cross entropy losses on the predictions
            epsilon = 10e-6
            float_labels = tf.cast(labels, tf.float32)
            float_labels_audio = tf.cast(labels_audio, tf.float32)
            cross_entropy_loss_audio = float_labels_audio * tf.log(predictions_audio + epsilon) + (
                                                                                                      1 - float_labels_audio) * tf.log(
                1 - predictions_audio + epsilon)
            cross_entropy_loss_audio = tf.negative(cross_entropy_loss_audio)
            cross_entropy_loss_frames = float_labels * tf.log(predictions_frames + epsilon) + (
                                                                                                  1 - float_labels) * tf.log(
                1 - predictions_frames + epsilon)
            cross_entropy_loss_frames = tf.negative(cross_entropy_loss_frames)

            cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_loss_audio, 1)) \
                                 + tf.reduce_mean(tf.reduce_sum(cross_entropy_loss_frames, 1))

            # Loss for the negative samples. We do not take into account the prediction loss in this case
            # The margin is not to focus too much on the negative samples


            #embedding_audio = math_ops.to_float(embedding_audio)
            #embedding_frames = math_ops.to_float(embedding_frames)
            # Cosine distance will be between 0 and 1 because the embeddings are greater than zero because of the relu

            radial_diffs = tf.multiply(embedding_audio, embedding_frames)
            cosine_distance = tf.reduce_sum(radial_diffs, 1, keep_dims=True) # The samples in the batch are separated
            cosine_distance_reversed = 1 - cosine_distance

            cosine_loss = tf.multiply(is_negative_float, 1/(1-margin)*tf.maximum(tf.constant(0.), cosine_distance - margin)) + \
                          tf.multiply(1-is_negative_float, cosine_distance_reversed)

            cosine_loss_mean = tf.reduce_mean(cosine_loss)

            total_loss = cosine_loss_mean + reg_lambda * tf.sqrt(cross_entropy_loss)

            return total_loss

class CrossEntropyLoss(BaseLoss):
    """Calculate the cross entropy loss between the predictions and labels.
    """

    def calculate_loss(self, predictions, labels, **unused_params):
        with tf.name_scope("loss_xent"):
            epsilon = 10e-6
            float_labels = tf.cast(labels, tf.float32)
            cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                1 - float_labels) * tf.log(1 - predictions + epsilon)
            cross_entropy_loss = tf.negative(cross_entropy_loss)
            return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class HingeLoss(BaseLoss):
    """Calculate the hinge loss between the predictions and labels.

    Note the subgradient is used in the backpropagation, and thus the optimization
    may converge slower. The predictions trained by the hinge loss are between -1
    and +1.
    """

    def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
        with tf.name_scope("loss_hinge"):
            float_labels = tf.cast(labels, tf.float32)
            all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
            all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
            sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
            hinge_loss = tf.maximum(
                all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
            return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class SoftmaxLoss(BaseLoss):
    """Calculate the softmax loss between the predictions and labels.

    The function calculates the loss in the following way: first we feed the
    predictions to the softmax activation function and then we calculate
    the minus linear dot product between the logged softmax activations and the
    normalized ground truth label.

    It is an extension to the one-hot label. It allows for more than one positive
    labels for each sample.
    """

    def calculate_loss(self, predictions, labels, **unused_params):
        with tf.name_scope("loss_softmax"):
            epsilon = 10e-8
            float_labels = tf.cast(labels, tf.float32)
            # l1 normalization (labels are no less than 0)
            label_rowsum = tf.maximum(
                tf.reduce_sum(float_labels, 1, keep_dims=True),
                epsilon)
            norm_float_labels = tf.div(float_labels, label_rowsum)
            softmax_outputs = tf.nn.softmax(predictions)
            softmax_loss = tf.negative(tf.reduce_sum(
                tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
        return tf.reduce_mean(softmax_loss)
