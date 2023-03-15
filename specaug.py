import numpy as np
import random

import myconfig

def freq_mask(spec, F=250, num_masks=1):
    test = spec.clone()
    num_mel_channels = test.shape[1]
    for i in range(0, num_masks):        
        freq = random.randrange(0, F)
        zero = random.randrange(0, num_mel_channels - freq)
        # avoids randrange error if values are equal and range is empty
        if (zero == zero + freq): 
            return test
        mask_end = random.randrange(zero, zero + freq) 
        test[0][zero:mask_end] = test.mean()
    return test

def time_mask(spec, time=40, num_masks=1):
    test = spec.clone()
    length = test.shape[2]
    for i in range(0, num_masks):
        t = random.randrange(0, time)
        zero = random.randrange(0, length - t)
        if (zero == zero + t): 
            return test
        mask_end = random.randrange(zero, zero + t)
        test[0][:,zero:mask_end] = test.mean()
    return test



def apply_specaug(features):
    """Apply SpecAugment to features."""
    seq_len, n_mfcc = features.shape
    outputs = features
    mean_feature = np.mean(features)

    # Frequancy masking.
    if random.random() < myconfig.SPECAUG_FREQ_MASK_PROB:
        width = random.randint(1, myconfig.SPECAUG_FREQ_MASK_MAX_WIDTH)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feature

    # Time masking.
    if random.random() < myconfig.SPECAUG_TIME_MASK_PROB:
        width = random.randint(1, myconfig.SPECAUG_TIME_MASK_MAX_WIDTH)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feature

    return outputs
