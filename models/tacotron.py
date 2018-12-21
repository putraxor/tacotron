import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from text.symbols import symbols
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet
from .rnn_wrappers import FrameProjection, StopProjection, TacotronDecoderWrapper
from .attention import LocationSensitiveAttention
from .custom_decoder import CustomDecoder


class Tacotron():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, f0_targets=None, sp_targets=None, ap_targets=None, stop_token_targets=None, global_step=None):
    '''Initializes the model for inference.

    Sets "mel_outputs", "linear_outputs", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
    '''
    with tf.variable_scope('inference') as scope:
      is_training = sp_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams
      target_depth = hp.num_f0 + hp.num_sp + hp.num_ap
      # decoder_targets = tf.concat([f0_targets, sp_targets, ap_targets], axis=-1)
      decoder_targets = tf.concat([tf.expand_dims(f0_targets, axis=-1), sp_targets, ap_targets], axis=-1)

      # Embeddings
      embedding_table = tf.get_variable(
        'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)                            # [N, T_in, embed_depth=256]

      # Encoder
      prenet_outputs = prenet(embedded_inputs, is_training, hp.prenet_depths)                      # [N, T_in, prenet_depths[-1]=128]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, hp.encoder_depth) # [N, T_in, encoder_depth=256]

      # Location sensitive attention
      attention_mechanism = LocationSensitiveAttention(hp.attention_depth, encoder_outputs)        # [N, T_in, attention_depth=256]

      # Decoder (layers specified bottom to top):
      multi_rnn_cell = MultiRNNCell([
          ResidualWrapper(GRUCell(hp.decoder_depth)),
          ResidualWrapper(GRUCell(hp.decoder_depth))
        ], state_is_tuple=True)                                                                    # [N, T_in, decoder_depth=256]

      # Frames Projection layer
      frame_projection = FrameProjection(target_depth * hp.outputs_per_step)                        # [N, T_out/r, M*r]

      # <stop_token> projection layer
      stop_projection = StopProjection(is_training, shape=hp.outputs_per_step)                     # [N, T_out/r, r]

      # Project onto r mel spectrograms (predict r outputs at each RNN step):
      decoder_cell = TacotronDecoderWrapper(is_training, attention_mechanism, multi_rnn_cell,
                                            frame_projection, stop_projection)

      if is_training:
        helper = TacoTrainingHelper(inputs, decoder_targets, target_depth, hp.outputs_per_step, global_step)
      else:
        helper = TacoTestHelper(batch_size, target_depth, hp.outputs_per_step)

      decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      (decoder_outputs, stop_token_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
         CustomDecoder(decoder_cell, helper, decoder_init_state),
         maximum_iterations=hp.max_iters)                                                          # [N, T_out/r, M*r]

      # Grab alignments from the final decoder state:
      alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

      # Reshape outputs to be one output per entry
      decoder_outputs = tf.reshape(decoder_outputs, [batch_size, -1, target_depth])      # [N, T_out, M]
      stop_token_outputs = tf.reshape(stop_token_outputs, [batch_size, -1])                                   # [N, T_out, M]

      # f0_outputs = decoder_outputs[:, :, 0]
      f0_outputs = tf.squeeze(decoder_outputs[:, :, 0])
      sp_outputs = decoder_outputs[:, :, 1 : 1 + hp.num_sp]
      ap_outputs = decoder_outputs[:, :, 1 + hp.num_sp:]

      # Add post-processing CBHG:
      # post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training, hp.postnet_depth)            # [N, T_out, postnet_depth=256]
      # linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)                                  # [N, T_out, F]

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.f0_outputs = f0_outputs
      self.sp_outputs = sp_outputs
      self.ap_outputs = ap_outputs
      self.stop_token_outputs = stop_token_outputs
      self.alignments = alignments
      self.f0_targets = f0_targets
      self.sp_targets = sp_targets
      self.ap_targets = ap_targets
      self.stop_token_targets = stop_token_targets
      log('Initialized Tacotron model. Dimensions: ')
      log('  embedding:               {}'.format(embedded_inputs.shape))
      log('  prenet out:              {}'.format(prenet_outputs.shape))
      log('  encoder out:             {}'.format(encoder_outputs.shape))
      log('  decoder out (r frames):  {}'.format(decoder_outputs.shape))
      log('  F0 out (1 frame):   {}'.format(f0_outputs.shape))
      log('  spectral envelop out (1 frame):   {}'.format(sp_outputs.shape))
      log('  aperioidicity out (1 frame):   {}'.format(ap_outputs.shape))
      #log('  postnet out:             {}'.format(post_outputs.shape))
      #log('  linear out:              {}'.format(linear_outputs.shape))
      log('  stop token:              {}'.format(stop_token_outputs.shape))


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      hp = self._hparams
      self.f0_loss = tf.reduce_mean(tf.abs(self.f0_targets - self.f0_outputs))
      self.sp_loss = tf.reduce_mean(tf.abs(self.sp_targets - self.sp_outputs))
      self.ap_loss = tf.reduce_mean(tf.abs(self.ap_targets - self.ap_outputs))
      self.stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                            labels=self.stop_token_targets,
                                            logits=self.stop_token_outputs))

      #l1 = tf.abs(self.linear_targets - self.linear_outputs)
      # Prioritize loss for frequencies under 4000 Hz.
      #n_priority_freq = int(4000 / (hp.sample_rate * 0.5) * hp.num_freq)
      #self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])

      # Compute the regularization weights
      reg_weight = 1e-6
      all_vars = tf.trainable_variables()
      self.regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
        if not('bias' in v.name or 'Bias' in v.name)]) * reg_weight

      #self.loss = self.mel_loss + self.linear_loss + self.stop_token_loss + self.regularization_loss
      self.loss = self.f0_loss + self.sp_loss + self.ap_loss + self.stop_token_loss + self.regularization_loss


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
