#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import librosa
import tensorflow as tf
import random
##########################################################################################
#
# 1. Weak Augmentation (Spectrogram base)
#
##########################################################################################

def weak_augment(image):
    aug_type = random.choice(range(4))

    ## 1) Left_Right_flip
    if aug_type == 0:
        image = tf.image.random_flip_left_right(image)
    
    ## 2) Random_brightness
    elif aug_type == 1:
        image = tf.image.random_brightness(image, max_delta=0.5)

    ## 3) Freq Masking
    elif aug_type == 2:
        image = freq_mask(image, mask_size_ratio=0.07)
    
    ## 4) Time Masking
    else:
        image = time_mask(image, mask_size_ratio=0.07)
    
    return image

        
def rolling_aug(spec, rolling_bound=1):
    time = np.shape(spec)[1]
    rolling_point = np.random.uniform(0,rolling_bound)

    left = spec[:,:int(rolling_point*time)]
    right = spec[:,int(rolling_point*time):]

    rolling_aug = np.concatenate((right, left), axis=1)

    return rolling_aug

def freq_mask(spec, mask_size_ratio=0.07):
    spec_aug = np.copy(spec)
    freq = np.shape(spec)[0]
    f_frame = np.random.randint(0, int(np.shape(spec)[0]*mask_size_ratio))
    f_location = np.random.randint(0,freq-f_frame)
    spec_aug[f_location: f_location+f_frame,:] = 0   
    return spec_aug

def time_mask(spec, mask_size_ratio=0.07):
    spec_aug = np.copy(spec)
    time = np.shape(spec)[1]
    t_frame = np.random.randint(0, int(np.shape(spec)[1]*mask_size_ratio))
    t_location = np.random.randint(0,time-t_frame)
    spec_aug[:,t_location:t_location+t_frame] = 0
    return spec_aug

##########################################################################################
#
# 2. Strong Augmentation (Audio signal base)
#
##########################################################################################

def strong_augment(signal, sr):
    aug_comb = np.random.uniform(0, 1, 4)
    while np.max(np.random.uniform(0, 1, 4)) <= 0.5:
        aug_comb = np.random.uniform(0, 1, 4)
    

    if aug_comb[0] > 0.5:
        rolling_strength = round(np.random.uniform(0.3, 0.7, 1)[0] * len(signal))
        signal = shifting(signal, rolling_strength)

    if aug_comb[1] > 0.5:
        rate = random.choice([0.8, 1.2])
        signal = stretching(signal, rate)

    if aug_comb[2] > 0.5:
        signal = pitch_shift(signal, sr = sr, n_steps = 6)
    
    if aug_comb[3] > 0.5:
        signal = white_noise(signal)
    
    return signal
    

## 1) White Noise add
def white_noise(audio_signal):
    wn = np.random.randn(len(audio_signal))
    audio_signal_wn = audio_signal + 0.005*wn
    return audio_signal_wn

## 2) Shifting
def shifting(audio_signal, rolling_strength):
    audio_signal_roll = np.roll(audio_signal, rolling_strength)
    return audio_signal_roll

## 3) Streching
def stretching(audio_signal, rate=1):
    input_length = len(audio_signal)
    audio_signal_stretched = librosa.effects.time_stretch(audio_signal, rate)
    if len(audio_signal_stretched)>input_length:
        audio_signal_stretched = audio_signal_stretched[:input_length]
    else:
        audio_signal_stretched = np.pad(audio_signal_stretched, (0, max(0, input_length - len(audio_signal_stretched))), "constant")
    return audio_signal_stretched

## 4) Pitch shift
def pitch_shift(audio_signal, sr, n_steps = 6):
    audio_signal_pitch_shifted = librosa.effects.pitch_shift(audio_signal, sr=sr, n_steps=n_steps)
    return audio_signal_pitch_shifted