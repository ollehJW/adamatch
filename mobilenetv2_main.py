# %%
import data_utils.augment
import data_utils.data_utils as data_utils
import data_utils.data_generator_set as dgs
from data_utils.spec_npy_gen import spec_to_npy
import json
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# %%
config = "./adamatch.params"
with open(config, 'r') as cfg:
    config = json.load(cfg)
    control_config = config['controls']
    parameter_config = config['parameters']

IMG_HEIGHT = 224
IMG_WIDTH = 224
SR = parameter_config['sr']
TIME_SIZE = parameter_config['time_size']
MIXUP_ALPHA = parameter_config['mixup_alpha']
MIXUP_BETA = parameter_config['mixup_beta']
CHANNEL_NAME = control_config['channel_name']
BATCH_SIZE = control_config['batch_size']

remove_file_list = ['20220210192037_M622GBB352_FLD165NCMA_54559202LP285B005202.tdms',
                    '20220221173902_M872MGG452_FLD165NBMA_57678202N4333B026302.tdms',
                    '20220207133321_M872AAA451_FLD165NBMA_54307202KU164B003802.tdms',
                    '20220207102952___75134202KP214B003201.tdms',
                    '20220221154237___57682202MZ407B026602.tdms',
                    '20220221132550___54567202MY514B025002.tdms',
                    '20220218105840___57683202MU171B026802.tdms',
                    '20220221092429___54559202MX737B024302.tdms',
                    '20220207133007___NoRead.tdms', '20220221080055___NoRead.tdms', 
                    '20220215080035___NoRead.tdms', '20220207103052___NoRead.tdms',
                    '20220209080038___NoRead.tdms', '20220208080127___NoRead.tdms',
                    '20220207104358___NoRead.tdms', '20220211080040___NoRead.tdms',
                    '20220207133025___NoRead.tdms', '20220214080034___NoRead.tdms'
                    ]


    

# %% make spectrogram with npy files
if os.path.exists(control_config['source_npy_dir']):
    print('npy is already created.')
else:
    spec_to_npy(control_config['source_data_dir'], control_config['source_npy_dir'], SR, TIME_SIZE, CHANNEL_NAME, IMG_HEIGHT, IMG_WIDTH, label_exist = True, remove_filename_list = [])
    spec_to_npy(control_config['target_data_dir'], control_config['target_npy_dir'], SR, TIME_SIZE, CHANNEL_NAME, IMG_HEIGHT, IMG_WIDTH, label_exist= False, remove_filename_list = remove_file_list)
    spec_to_npy(control_config['valid_data_dir'], control_config['valid_npy_dir'], SR, TIME_SIZE, CHANNEL_NAME, IMG_HEIGHT, IMG_WIDTH, label_exist = True, remove_filename_list = [])
# %%

# source dataset
file_name_list_sc, category_list_sc, file_label_list_sc = data_utils.prepare_data(
    train_path=control_config["source_npy_dir"], 
    remove_filename_list=control_config["remove_filename_list"])

use_index = list(np.where(file_label_list_sc[:, list(category_list_sc).index('OK')] == 1)[0])[:1000] + list(np.where(file_label_list_sc[:, list(category_list_sc).index('OK')] != 1)[0])
file_name_list_sc = [file_name_list_sc[idx] for idx in use_index]
file_label_list_sc = file_label_list_sc[use_index]

# target dataset
filelist = os.listdir((control_config["target_npy_dir"]))
file_name_list_tg = []
    
for file in filelist:
    file_name_list_tg.append(os.path.join(control_config["target_npy_dir"], file))
    
ok_samples = np.random.choice(list(np.where(file_label_list_sc[:, list(category_list_sc).index('OK')] == 1)[0]), len(file_name_list_tg))
file_label_list_tg = file_label_list_sc[ok_samples]
category_list_tg = category_list_sc

# validation dataset
file_name_list_vd, category_list_vd, file_label_list_vd = data_utils.prepare_data(
    train_path=control_config["valid_npy_dir"], 
    remove_filename_list=control_config["remove_filename_list"])



# %% Mix-up data index generation
mix_up_set_sc = data_utils.make_mixup_fileset(file_name_list_sc, category_list_sc, file_label_list_sc, seed=1004, mixup=True)
mix_up_set_tg = data_utils.make_mixup_fileset(file_name_list_tg, category_list_tg, file_label_list_tg, seed=1004, mixup=False)

# %%
sound_dataset_generator_sc = dgs.sound_dataset_generator(mix_up_set_sc, file_name_list_sc, file_label_list_sc, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, MIXUP_ALPHA, MIXUP_BETA, CHANNEL_NAME)
sound_dataset_generator_tg = dgs.sound_dataset_generator(mix_up_set_tg, file_name_list_tg, file_label_list_tg, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, MIXUP_ALPHA, MIXUP_BETA, CHANNEL_NAME)


# %%
#source_ds_w = tf.data.Dataset.from_generator(sound_dataset_generator_sc.weak_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))
#source_ds_s = tf.data.Dataset.from_generator(sound_dataset_generator_sc.strong_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))

source_ds_w = tf.data.Dataset.from_generator(sound_dataset_generator_sc.strong_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))
source_ds_s = tf.data.Dataset.from_generator(sound_dataset_generator_sc.weak_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))

#source_ds_w = source_ds_w.shuffle()
#source_ds_s = source_ds_s.shuffle()


source_ds_w = source_ds_w.batch(BATCH_SIZE)
source_ds_s = source_ds_s.batch(BATCH_SIZE)

final_source_ds = tf.data.Dataset.zip((source_ds_w, source_ds_s))

# %%
#target_ds_w = tf.data.Dataset.from_generator(sound_dataset_generator_tg.weak_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))
#target_ds_s = tf.data.Dataset.from_generator(sound_dataset_generator_tg.strong_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))

target_ds_w = tf.data.Dataset.from_generator(sound_dataset_generator_tg.strong_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))
target_ds_s = tf.data.Dataset.from_generator(sound_dataset_generator_tg.weak_aug_ds, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), (9)))

#target_ds_w = target_ds_w.shuffle()
#target_ds_s = target_ds_s.shuffle()


target_ds_w = target_ds_w.batch(BATCH_SIZE*3)
target_ds_s = target_ds_s.batch(BATCH_SIZE*3)

final_target_ds = tf.data.Dataset.zip((target_ds_w, target_ds_s))

            


# %%

EPOCHS = 20
STEPS_PER_EPOCH = len(mix_up_set_tg) // (3*BATCH_SIZE)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

AUTO = tf.data.AUTOTUNE
LEARNING_RATE = 0.000001

WEIGHT_DECAY = 0.0005
INIT = "he_normal"
DEPTH = 28
WIDTH_MULT = 2


# %%
def compute_loss_source(source_labels, logits_source_w, logits_source_s):
    loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
    # First compute the losses between original source labels and
    # predictions made on the weakly and strongly augmented versions
    # of the same images.
    w_loss = loss_func(source_labels, logits_source_w)
    s_loss = loss_func(source_labels, logits_source_s)
    return w_loss + s_loss


def compute_loss_target(target_pseudo_labels_w, logits_target_s, mask):
    loss_func = keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")
    target_pseudo_labels_w = tf.stop_gradient(target_pseudo_labels_w)
    # For calculating loss for the target samples, we treat the pseudo labels
    # as the ground-truth. These are not considered during backpropagation
    # which is a standard SSL practice.
    target_loss = loss_func(target_pseudo_labels_w, logits_target_s)

    # More on `mask` later.
    mask = tf.cast(mask, target_loss.dtype)
    target_loss *= mask
    return tf.reduce_mean(target_loss, 0)
# %%
def compute_mu(current_step, total_steps):
    pi = tf.constant(np.pi, dtype="float32")
    step = tf.cast(current_step, dtype="float32")
    return 0.5 - tf.cos(tf.math.minimum(pi, (2 * pi * step) / total_steps)) / 2

def train_step(source_w, source_s, source_labels, target_w, target_s, total_steps, tau = 0.9):
    
    combined_images = tf.concat([source_w, source_s, target_w, target_s], 0)
    combined_source = tf.concat([source_w, source_s], 0)

    total_source = tf.shape(combined_source)[0]
    total_target = tf.shape(tf.concat([target_w, target_s], 0))[0]

    with tf.GradientTape() as tape:
        ## Forward passes ##
        combined_logits = wrn_model(combined_images, training=True)
        z_d_prime_source = wrn_model(
            combined_source, training=False
        )  # No BatchNorm update.
        z_prime_source = combined_logits[:total_source]

        ## 1. Random logit interpolation for the source images ##
        lambd = tf.random.uniform((total_source, 9), 0, 1)
        final_source_logits = (lambd * z_prime_source) + (
            (1 - lambd) * z_d_prime_source
        )

        ## 2. Distribution alignment (only consider weakly augmented images) ##
        # Compute softmax for logits of the WEAKLY augmented SOURCE images.
        y_hat_source_w = tf.nn.softmax(final_source_logits[: tf.shape(source_w)[0]])

        # Extract logits for the WEAKLY augmented TARGET images and compute softmax.
        logits_target = combined_logits[total_source:]
        logits_target_w = logits_target[: tf.shape(target_w)[0]]
        y_hat_target_w = tf.nn.softmax(logits_target_w)

        # Align the target label distribution to that of the source.
        expectation_ratio = tf.reduce_mean(y_hat_source_w) / tf.reduce_mean(
            y_hat_target_w
        )
        y_tilde_target_w = tf.math.l2_normalize(
            y_hat_target_w * expectation_ratio, 1
        )

        ## 3. Relative confidence thresholding ##
        row_wise_max = tf.reduce_max(y_hat_source_w, axis=-1)
        final_sum = tf.reduce_mean(row_wise_max, 0)
        c_tau = tau * final_sum
        mask = tf.reduce_max(y_tilde_target_w, axis=-1) >= c_tau

        ## Compute losses (pay attention to the indexing) ##
        source_loss = compute_loss_source(
            source_labels,
            final_source_logits[: tf.shape(source_w)[0]],
            final_source_logits[tf.shape(source_w)[0] :],
        )
        target_loss = compute_loss_target(
            y_tilde_target_w, logits_target[tf.shape(target_w)[0] :], mask
        )

        t = compute_mu(current_step, total_steps)  # Compute weight for the target loss
        total_loss = source_loss + (t * target_loss)
        current_step.assign_add(
            1
        )  # Update current training step for the scheduler

    gradients = tape.gradient(total_loss, wrn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, wrn_model.trainable_variables))





    
# %%
IMG_SHAPE = (224, 224, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')
base_model.trainable = True
wrn_model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units=9)
])
print(f"Model has {wrn_model.count_params()/1e6} Million parameters.")
# %%
reduce_lr = keras.optimizers.schedules.CosineDecay(LEARNING_RATE, TOTAL_STEPS, 0.25)
optimizer = keras.optimizers.Adam(reduce_lr)
# wrn_model.compile(optimizer=optimizer)
current_step = tf.Variable(0, dtype="int64")
total_ds = tf.data.Dataset.zip((final_source_ds, final_target_ds))

sound_dataset_generator_vd = dgs.eval_dataset_load(file_name_list_vd, file_label_list_vd, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, CHANNEL_NAME)
target_ds_valid = tf.data.Dataset.from_generator(sound_dataset_generator_vd.eval_gen, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), ()))
target_ds_valid = target_ds_valid.batch(BATCH_SIZE)

sound_dataset_generator_sc = dgs.eval_dataset_load(file_name_list_sc, file_label_list_sc, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, CHANNEL_NAME)
source_ds_w_test = tf.data.Dataset.from_generator(sound_dataset_generator_sc.eval_gen, (tf.float32, tf.float64), ((IMG_HEIGHT, IMG_WIDTH, 3), ()))
source_ds_w_test = source_ds_w_test.batch(BATCH_SIZE)

#for epoch in range(EPOCHS):
#    print("Epoch: {} / {}".format(epoch, EPOCHS))
#    for (((source_w,source_label), (source_s, source_label)), ((target_w,target_label), (target_s, target_label))) in total_ds:
#        train_step(source_w, source_s, source_label, target_w, target_s, total_steps = TOTAL_STEPS)
#    
#    predicted = wrn_model.predict(target_ds_valid)
#    answer = np.argmax(predicted, axis = 1)
#    gt_list = np.argmax(file_label_list_vd, axis = 1)
#    distribution = confusion_matrix(gt_list, answer)
#    print(distribution)

for epoch in range(EPOCHS):
    print("Epoch: {} / {}".format(epoch, EPOCHS))
    for (((source_w,source_label), (source_s, source_label)), ((target_w,target_label), (target_s, target_label))) in tqdm(total_ds, total = STEPS_PER_EPOCH):
        train_step(source_w, source_s, source_label, target_w, target_s, total_steps = TOTAL_STEPS)
    
    print("Train")
    predicted = wrn_model.predict(source_ds_w_test)
    answer = np.argmax(predicted, axis = 1)
    gt_list = np.argmax(file_label_list_sc, axis = 1)
    distribution = confusion_matrix(gt_list, answer)
    print(distribution)
        
    print("Validation")
    predicted = wrn_model.predict(target_ds_valid)
    answer = np.argmax(predicted, axis = 1)
    gt_list = np.argmax(file_label_list_vd, axis = 1)
    distribution = confusion_matrix(gt_list, answer)
    print(distribution)



