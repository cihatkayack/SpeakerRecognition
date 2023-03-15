import librosa
import soundfile as sf
import random
import torch
import numpy as np

import myconfig
import dataset
import specaug


def extract_features(audio_file):
    """
    Extract MFCC features from an audio file thanks to librosa.
        shape=(TIME, MFCC).
        
    # MFCC(Mel Frequency Cepstral Coefficients)
    # MFCCs are a compact representation of the spectrum of an audio signal.
    # MFCC coefficients contain information about the rate changes in the different spectrum bands.
    # https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040
    """
    waveform, sample_rate = sf.read(audio_file)
    
    # Convert to mono-channel.
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())
        
    # Convert to 16kHz.
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr = sample_rate, target_sr=16000)
        
    features = librosa.feature.mfcc(
        y = waveform, sr = sample_rate, n_mfcc=myconfig.N_MFCC
        )
    
    return features.transpose()
   

def extract_sliding_windows(features):
    """
    Extract sliding windows from features
    """
    sliding_windows = []
    start_ind = 0
    
    while start_ind + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start_ind: start_ind + myconfig.SEQ_LEN, :])
        start_ind += myconfig.SLIDING_WINDOW_STEP
    
    return sliding_windows



def get_triplet_features(spk_to_utts):
    """
    Get a triplet of anchor/pos/neg features.
    utt = utterance
    """
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    # print("anchor",anchor_utt)
    # print("---------------------")
    # print("pos",pos_utt)
    # print("---------------------")
    # print("neg",neg_utt)
    return (extract_features(anchor_utt),
            extract_features(pos_utt),
            extract_features(neg_utt))


def trim_features(features, apply_specaug):
    
    length = features.shape[0]
    start_ind = random.randingt(0, length - myconfig.SEQ_LEN)
    
    trimmed_feature = features[start_ind:start_ind+myconfig.SEQ_LEN,:]
    
    if apply_specaug: # data augmentation
        trimmed_feature = specaug.apply_specaug(trimmed_feature)
    return trimmed_feature



class TrimmedTripletFeaturesFetcher:
    
    def __init__(self, spk_to_utts):
        self.spk_to_utts = spk_to_utts
        
    def __call__(self, _):
        anchor, pos, neg = get_triplet_features(self.spk_to_utts)
        
        while (anchor.shape[0] < myconfig.SEQ_LEN or
               pos.shape[0] < myconfig.SEQ_LEN or
               neg.shape[0] < myconfig.SEQ_LEN):
            anchor, pos, neg = get_triplet_features(self.spk_to_utts)
            
        return np.stack([trim_features(anchor, myconfig.SPECAUG_TRAINING),
                         trim_features(pos, myconfig.SPECAUG_TRAINING),
                         trim_features(neg, myconfig.SPECAUG_TRAINING)])



def get_batched_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batched triplet input for PyTorch."""
    fetcher = TrimmedTripletFeaturesFetcher(spk_to_utts)
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        input_arrays = pool.map(fetcher, range(batch_size))
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()
    return batch_input          


















     

