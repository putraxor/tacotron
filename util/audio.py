import librosa
import math
import numpy as np
import pyworld as vocoder
import soundfile as sf
import tensorflow as tf
from hparams import hparams as hp


def load_wav(path):
  wav, _ = sf.read(path)
  # rescale wav for unified measure for all clips
  return wav / np.abs(wav).max() * 0.999

def save_wav(wav, path):
  # rescaling for unified measure for all clips
  wav = wav / np.abs(wav).max() * 0.999
  # factor 0.5 in case of overflow for int16
  factor = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
  sf.write(path, wav, hp.sample_rate)

def trim_silence(wav):
  return librosa.effects.trim(wav, top_db= 60, frame_length=512, hop_length=128)[0]

def feature_extract(wav):
  fft_size = 2 * (hp.num_sp - 1)
  return vocoder.wav2world(wav, hp.sample_rate, fft_size)

def synthesize(f0, sp, ap):
  _f0 = np.float64(f0_denormalize(f0))
  _sp = np.float64(sp_denormalize(sp))
  _ap = np.float64(ap_denormalize(ap))
  return vocoder.synthesize(_f0, _sp, _ap, hp.sample_rate)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(x):
  # symmetric values
  return 2 * hp.max_abs_value * x - hp.max_abs_value

def _denormalize(x):
  # symmetric values
  return (x + hp.max_abs_value) / (2 * hp.max_abs_value)

def _denormalize_tensorflow(x):
  # symmetric values
  return (x + hp.max_abs_value) / (2 * hp.max_abs_value)

def f0_normalize(x):
  return _normalize(x / hp.max_f0_value)

def f0_denormalize(x):
  return _denormalize(x) * hp.max_f0_value

def f0_denormalize_tensorflow(x):
  return _denormalize_tensorflow(x) * hp.max_f0_value

def sp_normalize(x):
  # symmetric values
  return 2 * hp.max_abs_value * (_amp_to_db(x) + hp.ref_level_db) / (2 * hp.ref_level_db) - hp.max_abs_value

def sp_denormalize(x):
  # symmetric values
  return _db_to_amp((x + hp.max_abs_value) * (2 * hp.ref_level_db) / (2 * hp.max_abs_value) + hp.ref_level_db)

def sp_denormalize_tensorflow(x):
  # symmetric values
  return _db_to_amp_tensorflow((x + hp.max_abs_value) * (2 * hp.ref_level_db) / (2 * hp.max_abs_value) + hp.ref_level_db)

def ap_normalize(x):
  return _normalize(x / hp.max_ap_value)

def ap_denormalize(x):
  return _denormalize(x) * hp.max_ap_value

def ap_denormalize_tensorflow(x):
  return _denormalize_tensorflow(x) * hp.max_ap_value
