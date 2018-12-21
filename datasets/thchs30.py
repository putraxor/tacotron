from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import glob
from hparams import hparams as hp
import librosa
import soundfile as sf
import pyworld as vocoder

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the THCHS30 dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the THCHS30 dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  trn_files = glob.glob(os.path.join(in_dir, 'biaobei_48000', '*.trn'))

  for trn in trn_files:
    with open(trn) as f:
      pinyin = f.readline().strip('\n')
      wav_file = trn[:-4] + '.wav'
      task = partial(_process_utterance, out_dir, index, wav_file, pinyin)
      futures.append(executor.submit(task))
      index += 1
  return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(out_dir, index, wav_path, pinyin):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    pinyin: The pinyin of Chinese spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav, fs = sf.read(wav_path)

  # rescale wav for unified measure for all clips
  if np.abs(wav).max() > 1.0:
    wav = wav / np.abs(wav).max() * 0.999

  # trim silence
  wav = librosa.effects.trim(wav, top_db= 60, frame_length=512, hop_length=128)[0]

  # num_sp = fft_size // 2 + 1
  f0, sp, ap = vocoder.wav2world(wav, fs, 2 * (hp.num_sp - 1))
  n_frames = len(f0)
  if n_frames > hp.max_frame_num:
    return None

  # Write the spectrograms to disk:
  f0_filename = 'thchs30-f0-%05d.npy' % index
  sp_filename = 'thchs30-sp-%05d.npy' % index
  ap_filename = 'thchs30-ap-%05d.npy' % index
  np.reshape(f0, (-1, 1))
  np.save(os.path.join(out_dir, f0_filename), f0, allow_pickle=False)
  np.save(os.path.join(out_dir, sp_filename), sp, allow_pickle=False)
  np.save(os.path.join(out_dir, ap_filename), sp, allow_pickle=False)

  # Return a tuple describing this training example:
  return (f0_filename, sp_filename, ap_filename, n_frames, pinyin)
