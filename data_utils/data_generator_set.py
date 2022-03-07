import numpy as np
import librosa
from librosa.feature import melspectrogram
from librosa.display import specshow
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from .augment import weak_augment, strong_augment

class eval_dataset_load(object):
    def __init__(self, file_name_list, file_label_list, sr, img_height, img_width, time_size, channel_name):
        self.file_name_list = file_name_list
        self.file_label_list = file_label_list
        self.sr = sr
        self.img_height = img_height
        self.img_width = img_width
        self.time_size = time_size
        self.channel_name = channel_name
        self.label_answer = np.argmax(file_label_list, axis = 1)

    def eval_gen(self):
        for idx in range(len(self.file_name_list)):
            signal = get_signal(self.file_name_list[idx], self.sr, self.time_size, channel_name = self.channel_name)
            label = self.label_answer[idx]
            spec = get_specs(signal, self.sr, self.img_height, self.img_width)
            yield spec, label


class sound_dataset_generator(object):
    def __init__(self, mix_up_set, file_name_list, file_label_list, sr, img_height, img_width, time_size, mixup_alpha, mixup_beta, channel_name, is_train = True):
        self.mix_up_set = mix_up_set
        self.file_name_list = file_name_list
        self.file_label_list = file_label_list
        self.sr = sr
        self.img_height = img_height
        self.img_width = img_width
        self.time_size = time_size
        self.mixup_alpha = mixup_alpha
        self.mixup_beta =mixup_beta
        self.channel_name = channel_name
        self.is_train = is_train

    def weak_aug_ds(self):
        for idx in range(len(self.mix_up_set)):
            if isinstance(self.mix_up_set[idx], list):
                signal_ng = get_signal(self.file_name_list[self.mix_up_set[idx][0]], self.sr, self.time_size, channel_name = self.channel_name)
                signal_ok = get_signal(self.file_name_list[self.mix_up_set[idx][1]], self.sr, self.time_size, channel_name = self.channel_name)
                label_ng = self.file_label_list[self.mix_up_set[idx][0]]
                label_ok = self.file_label_list[self.mix_up_set[idx][1]]
                spec_ng = get_specs(signal_ng, self.sr, self.img_height, self.img_width)
                spec_ok = get_specs(signal_ok, self.sr, self.img_height, self.img_width)
                spec, label = mix_up_spec(spec_ng, spec_ok, label_ng, label_ok, self.mixup_alpha, self.mixup_beta)
            else:
                signal = get_signal(self.file_name_list[self.mix_up_set[idx]], self.sr, self.time_size, channel_name = self.channel_name)
                label = self.file_label_list[self.mix_up_set[idx]]
                spec = get_specs(signal, self.sr, self.img_height, self.img_width)
        
            if self.is_train:
                spec = weak_augment(spec)
        
            yield spec, label

    def strong_aug_ds(self):
        for idx in range(len(self.mix_up_set)):
            if isinstance(self.mix_up_set[idx], list):
                signal_ng = get_signal(self.file_name_list[self.mix_up_set[idx][0]], self.sr, self.time_size, channel_name = self.channel_name)
                signal_ok = get_signal(self.file_name_list[self.mix_up_set[idx][1]], self.sr, self.time_size, channel_name = self.channel_name)
                if self.is_train:
                    signal_ng = strong_augment(signal_ng, self.sr)
                    signal_ok = strong_augment(signal_ok, self.sr)
                label_ng = self.file_label_list[self.mix_up_set[idx][0]]
                label_ok = self.file_label_list[self.mix_up_set[idx][1]]
                spec_ng = get_specs(signal_ng, self.sr, self.img_height, self.img_width)
                spec_ok = get_specs(signal_ok, self.sr, self.img_height, self.img_width)
                spec, label = mix_up_spec(spec_ng, spec_ok, label_ng, label_ok, self.mixup_alpha, self.mixup_beta)
            else:
                signal = get_signal(self.file_name_list[self.mix_up_set[idx]], self.sr, self.time_size, channel_name = self.channel_name)
                if self.is_train:
                    signal = strong_augment(signal, self.sr)
                label = self.file_label_list[self.mix_up_set[idx]]
                spec = get_specs(signal, self.sr, self.img_height, self.img_width)

            yield spec, label






def mix_up_spec(spec1, spec2, spec1_label, spec2_label, mixup_alpha, mixup_beta):
    lam = np.random.beta(mixup_alpha, mixup_beta)
    lam = np.clip(lam, 0.00001, 0.99999)
    mixup_spec = (1-lam) * spec1 + lam * spec2
    mixup_label = (1-lam) * spec1_label + lam * spec2_label
    return (mixup_spec, mixup_label)


def get_specs(signal, sr, img_height, img_width):
    spec = mel_spectrogram(signal,sr,img_height,img_width)
    spec_img = spec_to_img(spec, n_channel=3)
    spec_img = standarize(spec_img, norm_axis="timefreq")
    return spec_img

def get_signal(fpath, sr, time_size, channel_name = 'CPsignal1'):
    if fpath.endswith('tdms'):
        signal = load_tdms(file_path=fpath, channel_name=channel_name)
    else:
        signal = load_wav(file_path=fpath, sr=sr)

    mid_signal = get_mid_signal(signal,sr,time_size)
    return mid_signal

def load_wav(file_path,sr):
    signal, _ = librosa.load(file_path, sr)
    return signal

def load_tdms(file_path, channel_name="CPsignal1"):
    """Load Single tdms file and returns target signal

    Parameters
    ----------
    file_path : str
        a single tdms file path
    channel_name : str
        Name of interest channel. Defaults to "CPsignal1".

    Returns
    --------
    np.array
        raw singal shape of (N,)
    """
    tdms_file = TdmsFile(file=file_path)
    tdms_group = tdms_file.groups()
    signal = tdms_group[0][channel_name][:]
    return signal





def get_mid_signal(signal, sr=25000, window_size=2.0):
    """get middle signal with a certain time length

    Parameters
    -----------
    signal : np.array
        original signal
    sr : int
        sampling ratio. Defaults to 25000.
    window_size : float
        time window size (seconds) of output. Defaults to 2.0.
    
    Raises
    ------
    AssertionError
        if original signal is less than window_size * sr

    Returns
    -------
    np.array
        middle segment of original signal.
    """
    segment_len = int(sr * window_size)
    if len(signal) < segment_len :
        raise ValueError("original signal is inadequate about your querries,"+
            "length of signal should be larger than or equal to sr * window_size")

    mid_idx = len(signal) // 2
    start_idx = mid_idx - int(np.ceil(segment_len / 2))
    start_idx = 0 if start_idx < 0 else start_idx
    end_idx = start_idx + segment_len
    return signal[start_idx:end_idx]

def mel_spectrogram(signal, sr, img_height, img_width):
    """Generate melody spectrogram with targeted size

    Parameters
    -----------
    signal : np.array
        1 dimensional (N,) signal
    sr : int
        sampling rate of signal
    img_height : int 
        target image height
    img_width : int 
        target image width

    Returns
    -------
    np.array
        melspectrogram. with shape of 2 dimensional numpy array
        (img_height, img_width)
    """
    hop_length = len(signal) // img_width
    n_fft = hop_length * 4 
    spec=melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                        hop_length=hop_length, n_mels=img_height)
    spec= librosa.power_to_db(S=spec, ref=np.max)
    return spec[:,:img_width]

def standarize(img, norm_axis="timefreq"):
    """Normalize a Single Image by Gaussian Distribution (mean, std).

    Parameters
    -----------
    img : np.array
        a single image np array with channel last (H x W x C) 
    norm_axis : str
        name of axis that will be applied standarization.
        norm_axis should be in ["height", "freq", "width", "time", "overall", "timefreq"]
    Returns
    -------
    np.array
        standarized image
    """
    if norm_axis in ['height', "freq"]:
        img = (img - np.mean(img, axis= 1)[:, np.newaxis ,:]) / \
             np.std(img, axis= 1)[: np.newaxis, :]
    elif norm_axis in ['width', "time"]:
        img = (img - np.mean(img, axis=0)[np.newaxis, :, :]) / \
             np.std(img, axis=0)[np.newaxis, :, :]
    elif norm_axis in ['overall', "timefreq"]:
        img = (img - np.mean(img, axis=(0,1))[np.newaxis, np.newaxis, :]) / \
             np.std(img, axis=(0,1))[np.newaxis, np.newaxis, :]
    return img

def spec_to_img(spec, n_channel=3):
    """make 2d np.array to 3d np.array, the last dimension works as a channel
    i.e. original 2d np.array spectrogram (H x W) -->
    image (H x W x C), where C is equal to n_channel

    Parameters
    ----------
    spec : np.array
        2 dimensional numpy array
    n_channel : int
        number of channel, it should be 1 or 3. Defaults to 3.

    Returns
    --------
    np.array 
        dimension increased numpy array, could work as image in 
        matplotlib or tensorflow image
    """
    assert n_channel in [1,3], ValueError("n_channel should be 1 or 3")
    img_1ch = spec.reshape(*spec.shape, 1)
    if n_channel ==1 :
        return img_1ch
    else:
        img_nch = np.concatenate([img_1ch]*n_channel, axis=-1)
        return img_nch 

def _plot_spec_for_dbg(save_fname, signal, sr=25000, img_width=224, img_height=224):
    """Plot and save mel spectrogram for debug

    Parameters
    -----------
    save_fname : str 
        path of your figure
    signal : np.array
        1 dimensional numpy array contains signal amplitude information
    sr : int
        sampling ratio of signal. Defaults to 25000
    img_width : int
        width of image/spectrogram. Defaults to 224.
    img_height : int
        height of image/spectrogram. Defaults to 224.
    """
    plt.figure()
    spec = mel_spectrogram(signal, sr, img_width=img_width, img_height=img_height)
    img = specshow(spec, sr=sr, hop_length=len(signal)/img_width, x_axis='time')
    save_fname = save_fname.rstrip('.png')
    plt.savefig(f"{save_fname}.png")