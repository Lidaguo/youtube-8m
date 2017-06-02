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

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
from tensorflow.python.ops import io_ops
from tensorflow import logging
from tensorflow.python.framework import dtypes


def resize_axis(tensor, axis, new_size, fill_value=0):
    """Truncates or pads a tensor to new_size on on a given axis.

    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.

    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.

    Returns:
      The resized tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)

    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized


class BaseReader(object):
    """Inherit from this class when implementing new readers."""

    def prepare_reader(self, unused_filename_queue):
        """Create a thread for generating prediction and label tensors."""
        raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
    """Reads TFRecords of pre-aggregated Examples.

    The TFRecords must contain Examples with a sparse int64 'labels' feature and
    a fixed length float32 feature, obtained from the features in 'feature_name'.
    The float features are assumed to be an average of dequantized values.
    """

    def __init__(self,
                 num_classes=4716,
                 feature_sizes=[1024],
                 feature_names=["mean_inc3"],
                 random_selection=0,
                 negative_sampling=False,
                 percentage_negative=0.8):
        """Construct a YT8MAggregatedFeatureReader.

        Args:
          num_classes: a positive integer for the number of classes.
          feature_sizes: positive integer(s) for the feature dimensions as a list.
          feature_names: the feature name(s) in the tensorflow record as a list.
        """

        assert len(feature_names) == len(feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format( \
                len(feature_names), len(feature_sizes))

        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.random_selection = random_selection
        self.negative_sampling = negative_sampling
        self.percentage_negative = percentage_negative

    def prepare_reader(self, filename_queue, batch_size=1024):
        """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

        Args:
          filename_queue: A tensorflow queue of filename locations.
          random_selection: An int detailing which kind of selection among features has to be done
            0: normal -> all the features concatenated
            1: the specified features with the non-specified put to zero (always): for evaluation
            2: a random selection. 1/3 normal, 1/3 only audio (frames to zero) and 1/3 only frames (audio to zero)
                (for training)
            This implies the code slightly hardcoded (size of features) because it will NOT depend on the command line

        Returns:
          A tuple of video indexes, features, labels, and padding data.
        """
        reader = tf.TFRecordReader()
        logging.info("Batch: " + str(batch_size))
        _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

        tf.add_to_collection("serialized_examples", serialized_examples)
        return self.prepare_serialized_examples(serialized_examples)

    def prepare_serialized_examples(self, serialized_examples):

        logging.set_verbosity(tf.logging.DEBUG)

        # hardcoded values
        len_features_frames = 1024
        len_features_audio = 128
        name_frames = "mean_rgb"
        name_audio = "mean_audio"

        # set the mapping from the fields to data types in the proto
        num_features = len(self.feature_names)
        assert num_features > 0, "self.feature_names is empty!"
        assert len(self.feature_names) == len(self.feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format( \
                len(self.feature_names), len(self.feature_sizes))

        feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                       "labels": tf.VarLenFeature(tf.int64)}
        logging.debug("self.random_selection es " + str(self.random_selection))

        zeros_float = tf.zeros([tf.shape(serialized_examples)[0]])
        # Manera cutre de crear un vector de False. Alguna altra manera ha d'haver-hi
        is_negative = tf.not_equal(zeros_float, zeros_float)

        for feature_index in range(num_features):
            feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
                [self.feature_sizes[feature_index]], tf.float32)

        features = tf.parse_example(serialized_examples, features=feature_map)
        features_rgb = features[name_frames]
        features_audio = features[name_audio]

        labels_audio = tf.sparse_to_indicator(features["labels"], self.num_classes)

        batch_size = tf.shape(features[name_frames])[0]

        if self.negative_sampling:

            labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
            labels.set_shape([None, self.num_classes])

            def return_itself(a, b):
                return a, b

            # 80% of the samples are negative
            number_neg_sample = tf.random_uniform([]
                                                  , minval=0., maxval=1., dtype=tf.float32,
                                                  name="random_number_neg_sample")
            constant = tf.constant(self.percentage_negative)
            batch_size = tf.shape(features_rgb)[0]
            logging.info("-----------------")
            logging.info(batch_size)
            is_negative = tf.random_uniform([batch_size,1] , minval=0, maxval=1)
            is_negative = tf.less(is_negative, constant)
            features_audio_return, labels_audio = self.sample_negatively(features, labels, is_negative)
            concatenated_features = tf.concat([ features_rgb, features_audio_return],1)

        else:
            # Normal case, leave as it was
            # We can use python comparisons because they are checked only when creating the graph
            if self.random_selection == 0 | (self.random_selection == 1 & num_features > 1):
                for feature_index in range(num_features):
                    feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
                        [self.feature_sizes[feature_index]], tf.float32)

                features = tf.parse_example(serialized_examples, features=feature_map)

                labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
                labels.set_shape([None, self.num_classes])
                labels_audio = labels
                concatenated_features = tf.concat([
                    features[feature_name] for feature_name in self.feature_names], 1)

            # Evaluation with only one of the two features
            elif self.random_selection == 1:
                feature_map[name_frames] = tf.FixedLenFeature([len_features_frames], tf.float32)
                feature_map[name_audio] = tf.FixedLenFeature([len_features_audio], tf.float32)

                features = tf.parse_example(serialized_examples, features=feature_map)

                labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
                labels.set_shape([None, self.num_classes])

                # In this point there is only 1 feature_name
                # We can use python comparisons because they are checked only when creating the graph
                if self.feature_names[0] == name_frames:
                    concatenated_features = tf.concat([features[name_frames], tf.zeros_like(features[name_audio])], 1)
                else:
                    concatenated_features = tf.concat([tf.zeros_like(features[name_frames]), features[name_audio]], 1)

            # Training with thirds
            else:
                feature_map[name_frames] = tf.FixedLenFeature([len_features_frames], tf.float32)
                feature_map[name_audio] = tf.FixedLenFeature([len_features_audio], tf.float32)

                features = tf.parse_example(serialized_examples, features=feature_map)

                labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
                labels.set_shape([None, self.num_classes])
                number = tf.random_uniform([], minval=0., maxval=3., dtype=tf.float32, name="random_number")

                features_rgb = features[name_frames]
                features_audio = features[name_audio]

                one = tf.constant(1.)
                two = tf.constant(2.)

                features_audio = tf.cond(tf.less(number, one), lambda: tf.clip_by_value(features_audio, 0, 0),
                                         lambda: features_audio)
                features_rgb = tf.cond(tf.greater(number, two), lambda: tf.clip_by_value(features_rgb, 0, 0),
                                       lambda: features_rgb)

                concatenated_features = tf.concat([features_rgb, features_audio], 1, name="concat_features")

        return features["video_id"], concatenated_features, labels, tf.ones(
            [tf.shape(serialized_examples)[0]]), is_negative, labels_audio

    def sample_negatively(self, features, labels, is_negative):
        features_audio = features["mean_audio"]
        number_labels = tf.shape(labels)[1]
        batch_size = tf.shape(labels)[0]
        # shuffled = tf.random_shuffle(tf.concat([features_audio, labels],1))
        # features_audio_shuffled = shuffled[:,0:128]
        # labels_shuffled = shuffled[:, 128:128+number_labels]

        index_out = tf.constant(0)
        found_out_out = tf.constant(0)
        i2 = tf.constant(0)

        def cond1(index, features_audio, features_audio_return, index2_out, found_out, labels_audio_return):
            return tf.less(index, batch_size)

        def body(index, features_audio, features_audio_return, index2_out, found_out, labels_audio_return):
            # Check if the new feature the has all labels different
            # If not, take the next one until some has all different.
            # TODO: Control the case where none of the features in the batch has all the labels different.
            # (Nearly impossible for batches greater than 10 or so)

            def cond2(index2, found):
                return tf.equal(found, tf.constant(0))

            def body2(index2, found):
                index2 = tf.random_uniform([], minval=0, maxval=batch_size, dtype=tf.int32)
                # Look if there is any common label
                shared_classes = tf.logical_and(labels[index, :], labels[index2, :])
                found_boolean = tf.reduce_any(shared_classes)
                found = tf.cond(found_boolean, lambda: tf.constant(0), lambda: tf.constant(1))
                return index2, found

            index2_out = tf.while_loop(cond2, body2, [index2_out, found_out], name="index2_out2")[0]

            features_audio_2 = tf.cond(is_negative[index,0],
                                       lambda: tf.reshape(features_audio[index2_out, :], [1,128]),
                                       lambda: tf.reshape(features_audio[index, :],[1,128]))
            features_audio_return = tf.concat([features_audio_return, features_audio_2], 0,
                                              name="features_audio_return")

            labels_audio_2 = tf.cond(is_negative[index,0],
                                     lambda: tf.reshape(labels[index2_out, :], [1, 4716]),
                                     lambda: tf.reshape(labels[index, :], [1, 4716]))
            labels_audio_return = tf.concat([labels_audio_return, labels_audio_2], 0, name="labels_audio_return")
            index = tf.add(index, 1)  # Update index

            return index, features_audio, features_audio_return, index2_out, found_out, labels_audio_return

        features_audio_return_out = tf.ones([1, 128], name="features_audio_return")
        labels_audio_return_out = tf.equal(tf.ones([1, 4716]),
                                           tf.ones([1, 4716]))  # Manera cutre de inicialitzar amb bools
        variables = tf.while_loop(cond1, body, [index_out, features_audio, features_audio_return_out, i2,
                                                found_out_out, labels_audio_return_out],
                                  shape_invariants=[index_out.get_shape(), features_audio.get_shape(),
                                                    tf.TensorShape([None, 128]), i2.get_shape(),
                                                    found_out_out.get_shape(), tf.TensorShape([None, 4716])],
                                  name="while1")
        feat_audio = variables[2]
        labels_audio = variables[5]

        return feat_audio[1:batch_size + 1, :], labels_audio[1:batch_size + 1,
                                                :]  # We delete the first row because it is the "ones" to initialize


class YT8MFrameFeatureReader(BaseReader):
    """Reads TFRecords of SequenceExamples.

    The TFRecords must contain SequenceExamples with the sparse in64 'labels'
    context feature and a fixed length byte-quantized feature vector, obtained
    from the features in 'feature_names'. The quantized features will be mapped
    back into a range between min_quantized_value and max_quantized_value.
    """

    def __init__(self,
                 num_classes=4716,
                 feature_sizes=[1024],
                 feature_names=["inc3"],
                 max_frames=300):
        """Construct a YT8MFrameFeatureReader.

        Args:
          num_classes: a positive integer for the number of classes.
          feature_sizes: positive integer(s) for the feature dimensions as a list.
          feature_names: the feature name(s) in the tensorflow record as a list.
          max_frames: the maximum number of frames to process.
        """

        assert len(feature_names) == len(feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format(
                len(feature_names), len(feature_sizes))

        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.max_frames = max_frames

    def get_video_matrix(self,
                         features,
                         feature_size,
                         max_frames,
                         max_quantized_value,
                         min_quantized_value):
        """Decodes features from an input string and quantizes it.

        Args:
          features: raw feature values
          feature_size: length of each frame feature vector
          max_frames: number of frames (rows) in the output feature_matrix
          max_quantized_value: the maximum of the quantized value.
          min_quantized_value: the minimum of the quantized value.

        Returns:
          feature_matrix: matrix of all frame-features
          num_frames: number of frames in the sequence
        """
        decoded_features = tf.reshape(
            tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
            [-1, feature_size])

        num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
        feature_matrix = utils.Dequantize(decoded_features,
                                          max_quantized_value,
                                          min_quantized_value)
        feature_matrix = resize_axis(feature_matrix, 0, max_frames)
        return feature_matrix, num_frames

    def prepare_reader(self,
                       filename_queue,
                       max_quantized_value=2,
                       min_quantized_value=-2,
                       random_selection=0,
                       negative_sampling=False):
        """Creates a single reader thread for YouTube8M SequenceExamples.

        Args:
          filename_queue: A tensorflow queue of filename locations.
          max_quantized_value: the maximum of the quantized value.
          min_quantized_value: the minimum of the quantized value.

        Returns:
          A tuple of video indexes, video features, labels, and padding data.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        return self.prepare_serialized_examples(serialized_example,
                                                max_quantized_value, min_quantized_value)

    def prepare_serialized_examples(self, serialized_example,
                                    max_quantized_value=2, min_quantized_value=-2):

        contexts, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={"video_id": tf.FixedLenFeature(
                [], tf.string),
                "labels": tf.VarLenFeature(tf.int64)},
            sequence_features={
                feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
                for feature_name in self.feature_names
                })

        # read ground truth labels
        labels = (tf.cast(
            tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
                               validate_indices=False),
            tf.bool))

        # loads (potentially) different types of features and concatenates them
        num_features = len(self.feature_names)
        assert num_features > 0, "No feature selected: feature_names is empty!"

        assert len(self.feature_names) == len(self.feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format( \
                len(self.feature_names), len(self.feature_sizes))

        num_frames = -1  # the number of frames in the video
        feature_matrices = [None] * num_features  # an array of different features
        for feature_index in range(num_features):
            feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
                features[self.feature_names[feature_index]],
                self.feature_sizes[feature_index],
                self.max_frames,
                max_quantized_value,
                min_quantized_value)
            if num_frames == -1:
                num_frames = num_frames_in_this_feature
            else:
                tf.assert_equal(num_frames, num_frames_in_this_feature)

            feature_matrices[feature_index] = feature_matrix

        # cap the number of frames at self.max_frames
        num_frames = tf.minimum(num_frames, self.max_frames)

        # concatenate different features
        video_matrix = tf.concat(feature_matrices, 1)

        # convert to batch format.
        # TODO: Do proper batch reads to remove the IO bottleneck.
        batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
        batch_video_matrix = tf.expand_dims(video_matrix, 0)
        batch_labels = tf.expand_dims(labels, 0)
        batch_frames = tf.expand_dims(num_frames, 0)

        return batch_video_ids, batch_video_matrix, batch_labels, batch_frames, [], []
