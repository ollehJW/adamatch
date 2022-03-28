from .data_generator_set import get_signal, get_specs
import glob
import os
import numpy as np

def spec_to_npy(source_dir, target_dir, sr, time_size, channel_name, img_height, img_width, label_exist = True, remove_filename_list = []):
    os.makedirs(target_dir, exist_ok=True)

    if label_exist:
        classes = os.listdir(source_dir)
        for class_ in classes :
            os.makedirs(os.path.join(target_dir, class_), exist_ok=True)

        for cls_ in classes: 
            for src_path in glob.glob(os.path.join(source_dir, cls_)+"/*.tdms"):
                target_fname = os.path.basename(src_path)
                if target_fname in remove_filename_list:
                    print("{} is removed.".format(target_fname))
                else:
                    signal = get_signal(src_path, sr, time_size, channel_name = channel_name)
                    spec = get_specs(signal, sr, img_height, img_width)
                    target_fname=target_fname.split(".")[0] + ".npy"
                    target_path = os.path.join(os.path.join(target_dir, cls_)+"/"+target_fname)
                    np.save(target_path, spec)

            for src_path in glob.glob(os.path.join(source_dir, cls_)+"/*.wav"):
                target_fname = os.path.basename(src_path)
                if target_fname in remove_filename_list:
                    print("{} is removed.".format(target_fname))
                else:
                    signal = get_signal(src_path, sr, time_size, channel_name = channel_name)
                    spec = get_specs(signal, sr, img_height, img_width)
                    target_fname=target_fname.split(".")[0] + ".npy"
                    target_path = os.path.join(os.path.join(target_dir, cls_)+"/"+target_fname)
                    np.save(target_path, spec)
    else:
        for src_path in glob.glob(source_dir+"*.tdms"):
            target_fname = os.path.basename(src_path)
            if target_fname in remove_filename_list:
                    print("{} is removed.".format(target_fname))
            else:
                signal = get_signal(src_path, sr, time_size, channel_name = channel_name)
                spec = get_specs(signal, sr, img_height, img_width)
                target_fname=target_fname.split(".")[0] + ".npy"
                target_path = os.path.join(target_dir + target_fname)
                np.save(target_path, spec)

        for src_path in glob.glob(source_dir+"*.wav"):
            target_fname = os.path.basename(src_path)
            if target_fname in remove_filename_list:
                    print("{} is removed.".format(target_fname))
            else:
                signal = get_signal(src_path, sr, time_size, channel_name = channel_name)
                spec = get_specs(signal, sr, img_height, img_width)
                target_fname=target_fname.split(".")[0] + ".npy"
                target_path = os.path.join(target_dir + target_fname)
                np.save(target_path, spec)
