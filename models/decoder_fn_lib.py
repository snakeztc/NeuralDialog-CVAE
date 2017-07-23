#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def context_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                 start_of_sequence_id, end_of_sequence_id,
                                 maximum_length, num_decoder_symbols, context_vector,
                                 dtype=dtypes.int32, name=None, decode_type='greedy'):
    """ Simple decoder function for a sequence-to-sequence model used in the
      `dynamic_rnn_decoder`.

      Args:
        output_fn: An output function to project your `cell_output` onto class
        logits.

        If `None` is supplied it will act as an identity function, which
        might be wanted when using the RNNCell `OutputProjectionWrapper`.

        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
        embeddings: The embeddings matrix used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        context_vector: an extra vector that should be appended to the input embedding
        dtype: (default: `dtypes.int32`) The default data type to use when
        handling integer objects.
        name: (default: `None`) NameScope for the decoder function;
          defaults to "simple_decoder_fn_inference"

      Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for inference.
      """
    with ops.name_scope(name, "simple_decoder_fn_inference",
                        [output_fn, encoder_state, embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         maximum_length, num_decoder_symbols, dtype]):
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maxium_length_int = maximum_length + 1
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """ Decoder function used in the `dynamic_rnn_decoder` with the purpose of
            inference.

            The main difference between this decoder function and the `decoder_fn` in
            `simple_decoder_fn_train` is how `next_cell_input` is calculated. In this
            decoder function we calculate the next input by applying an argmax across
            the feature dimension of the output from the decoder. This is a
            greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
            use beam-search instead.

            Args:
              time: positive integer constant reflecting the current timestep.
              cell_state: state of RNNCell.
              cell_input: input provided by `dynamic_rnn_decoder`.
              cell_output: output of RNNCell.
              context_state: context state provided by `dynamic_rnn_decoder`.

            Returns:
              A tuple (done, next state, next input, emit output, next context state)
              where:

              done: A boolean vector to indicate which sentences has reached a
              `end_of_sequence_id`. This is used for early stopping by the
              `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
              all elements as `true` is returned.

              next state: `cell_state`, this decoder function does not modify the
              given state.

              next input: The embedding from argmax of the `cell_output` is used as
              `next_input`.

              emit output: If `output_fn is None` the supplied `cell_output` is
              returned, else the `output_fn` is used to update the `cell_output`
              before calculating `next_input` and returning `cell_output`.

              next context state: `context_state`, this decoder function does not
              modify the given context state. The context state could be modified when
              applying e.g. beam search.
        """
        with ops.name_scope(name, "simple_decoder_fn_inference",
                            [time, cell_state, cell_input, cell_output,
                             context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones([batch_size, ], dtype=dtype) * (
                    start_of_sequence_id)
                done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros([num_decoder_symbols],
                                              dtype=dtypes.float32)
                context_state = tf.zeros((batch_size, maxium_length_int), dtype=tf.int32)
            else:
                cell_output = output_fn(cell_output)

                if decode_type == 'sample':
                    matrix_U = -1.0 * tf.log(
                        -1.0 * tf.log(tf.random_uniform(tf.shape(cell_output), minval=0.0, maxval=1.0)))
                    next_input_id = math_ops.cast(
                        tf.argmax(tf.subtract(cell_output, matrix_U), dimension=1), dtype=dtype)
                elif decode_type == 'greedy':
                    next_input_id = math_ops.cast(
                        math_ops.argmax(cell_output, 1), dtype=dtype)
                else:
                    raise ValueError("unknown decode type")

                done = math_ops.equal(next_input_id, end_of_sequence_id)
                # save the results into context state
                expanded_next_input = tf.expand_dims(next_input_id, axis=1)
                sliced_context_state = tf.slice(context_state, [0, 0], [-1, maxium_length_int - 1])
                context_state = tf.concat([expanded_next_input, sliced_context_state], axis=1)
                context_state = tf.reshape(context_state, [batch_size, maxium_length_int])

            next_input = array_ops.gather(embeddings, next_input_id)
            if context_vector is not None:
                next_input = tf.concat([next_input, context_vector], axis=1)
            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
                                         lambda: array_ops.ones([batch_size, ], dtype=dtypes.bool),
                                         lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def context_decoder_fn_train(encoder_state, context_vector, name=None):
    with ops.name_scope(name, "simple_decoder_fn_train", [encoder_state]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """ Decoder function used in the `dynamic_rnn_decoder` with the purpose of
            training.

            Args:
              time: positive integer constant reflecting the current timestep.
              cell_state: state of RNNCell.
              cell_input: input provided by `dynamic_rnn_decoder`.
              cell_output: output of RNNCell.
              context_state: context state provided by `dynamic_rnn_decoder`.

            Returns:
              A tuple (done, next state, next input, emit output, next context state)
              where:

              done: `None`, which is used by the `dynamic_rnn_decoder` to indicate
              that `sequence_lengths` in `dynamic_rnn_decoder` should be used.

              next state: `cell_state`, this decoder function does not modify the
              given state.

              next input: `cell_input`, this decoder function does not modify the
              given input. The input could be modified when applying e.g. attention.

              emit output: `cell_output`, this decoder function does not modify the
              given output.

              next context state: `context_state`, this decoder function does not
              modify the given context state. The context state could be modified when
              applying e.g. beam search.
          """
        with ops.name_scope(name, "simple_decoder_fn_train",
                            [time, cell_state, cell_input, cell_output,
                             context_state]):
            if context_vector is not None:
                cell_input = tf.concat([cell_input, context_vector], axis=1)
            if cell_state is None:  # first call, return encoder_state
                return (None, encoder_state, cell_input, cell_output, context_state)
            else:
                return (None, cell_state, cell_input, cell_output, context_state)

    return decoder_fn
