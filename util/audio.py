import librosa
import math
import numpy as np
import pysptk
import pyworld as vocoder
import soundfile as sf
import tensorflow as tf
from hparams import hparams as hp


def load_wav(path):
  wav, _ = sf.read(path)
  return wav

def save_wav(wav, path):
  # rescaling for unified measure for all clips
  # wav = wav / np.abs(wav).max() * 0.999
  sf.write(path, wav, hp.sample_rate)

def trim_silence(wav):
  return librosa.effects.trim(wav, top_db= 60, frame_length=512, hop_length=128)[0]

def feature_extract(wav):
  return vocoder.wav2world(wav, hp.sample_rate, hp.fft_size, ap_depth=hp.n_ap)

def synthesize(lf0, mgc, bap):
  lf0 = np.where(lf0 < 1, 0.0, lf0)
  f0 = f0_denormalize(lf0)
  sp = sp_denormalize(mgc)
  ap = ap_denormalize(bap, lf0)
  wav = vocoder.synthesize(f0, sp, ap, hp.sample_rate)
  return wav

def f0_normalize(x):
  return np.log(np.where(x == 0.0, 1.0, x)).astype(np.float32)

def f0_denormalize(x):
  return np.where(x == 0.0, 0.0, np.exp(x.astype(np.float64)))

def sp_normalize(x):
  sp = 32768.0 * np.sqrt(x)
  return pysptk.sptk.mcep(sp.astype(np.float32), order=hp.n_mgc - 1, alpha=hp.mcep_alpha, maxiter=0,
                          threshold=0.001, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

def sp_denormalize(x):
  sp = pysptk.sptk.mgc2sp(x.astype(np.float64), order=hp.n_mgc - 1, alpha=hp.mcep_alpha, gamma=0.0, fftlen=hp.fft_size)
  return np.square(sp / 32768.0)

def ap_normalize(x):
  return x.astype(np.float32)

def ap_denormalize(x, lf0):
  for i in range(len(lf0)):
    x[i] = np.where(lf0[i] == 0, np.zeros(x.shape[1]), x[i])
  return x.astype(np.float64)
