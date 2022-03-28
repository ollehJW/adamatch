# %%
import data_utils.augment
import data_utils.data_utils as data_utils
import data_utils.data_generator_set as dgs
import json
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# %%
config = "./adamatch.params"
with open(config, 'r') as cfg:
    config = json.load(cfg)
    control_config = config['controls']
    parameter_config = config['parameters']

IMG_HEIGHT = 32
IMG_WIDTH = 32
SR = parameter_config['sr']
TIME_SIZE = parameter_config['time_size']
MIXUP_ALPHA = parameter_config['mixup_alpha']
MIXUP_BETA = parameter_config['mixup_beta']
CHANNEL_NAME = control_config['channel_name']
BATCH_SIZE = control_config['batch_size']

# %%
parameter_config
# %%

# set data generation by parameters
file_name_list_sc, category_list_sc, file_label_list_sc = data_utils.prepare_data(
    train_path=control_config["source_data_dir"], 
    remove_filename_list=control_config["remove_filename_list"])

use_index = list(np.where(file_label_list_sc[:, list(category_list_sc).index('OK')] == 1)[0])[:2000] + list(np.where(file_label_list_sc[:, list(category_list_sc).index('OK')] != 1)[0])
file_name_list_sc = [file_name_list_sc[idx] for idx in use_index]
file_label_list_sc = file_label_list_sc[use_index]

file_name_list_tg, category_list_tg, file_label_list_tg = data_utils.prepare_data(
    train_path=control_config["target_data_dir"], 
    remove_filename_list=control_config["remove_filename_list"])

tg_list = list(np.where(file_label_list_tg[:, list(category_list_tg).index('OK')] == 1)[0]) + list(np.where(file_label_list_tg[:, list(category_list_tg).index('NG_support_spring')] == 1)[0])
file_name_list_tg = [file_name_list_tg[i] for i in tg_list]
file_label_list_tg = file_label_list_tg[tg_list]

file_name_list_tg_test = [file_name_list_tg[idx] for idx in list(range(2000,4000)) + [4004]]
file_label_list_tg_test = file_label_list_tg[list(range(2000,4000)) + [4004]]

file_name_list_tg = [file_name_list_tg[idx] for idx in list(range(2000)) + [4000,4001,4002,4003]]
file_label_list_tg = file_label_list_tg[list(range(2000)) + [4000,4001,4002,4003]]


# %% Mix-up data index generation
mix_up_set_sc = data_utils.make_mixup_fileset(file_name_list_sc, category_list_sc, file_label_list_sc, seed=1004, mixup=True)
mix_up_set_tg = data_utils.make_mixup_fileset(file_name_list_tg, category_list_tg, file_label_list_tg, seed=1004, mixup=True)




# %%
sound_dataset_generator_sc = dgs.sound_dataset_generator(mix_up_set_sc, file_name_list_sc, file_label_list_sc, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, MIXUP_ALPHA, MIXUP_BETA, CHANNEL_NAME)
sound_dataset_generator_tg = dgs.sound_dataset_generator(mix_up_set_tg, file_name_list_tg, file_label_list_tg, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, MIXUP_ALPHA, MIXUP_BETA, CHANNEL_NAME)


# %%
source_ds_w = tf.data.Dataset.from_generator(sound_dataset_generator_sc.weak_aug_ds, (tf.float32, tf.float64), ((224, 224, 3), (9)))
source_ds_s = tf.data.Dataset.from_generator(sound_dataset_generator_sc.strong_aug_ds, (tf.float32, tf.float64), ((224, 224, 3), (9)))

source_ds_w = source_ds_w.batch(BATCH_SIZE)
source_ds_s = source_ds_s.batch(BATCH_SIZE)

final_source_ds = tf.data.Dataset.zip((source_ds_w, source_ds_s))

# %%
target_ds_w = tf.data.Dataset.from_generator(sound_dataset_generator_tg.weak_aug_ds, (tf.float32, tf.float64), ((224, 224, 3), (9)))
target_ds_s = tf.data.Dataset.from_generator(sound_dataset_generator_tg.strong_aug_ds, (tf.float32, tf.float64), ((224, 224, 3), (9)))

target_ds_w = target_ds_w.batch(BATCH_SIZE*3)
target_ds_s = target_ds_s.batch(BATCH_SIZE*3)

final_target_ds = tf.data.Dataset.zip((target_ds_w, target_ds_s))

            


# %%

EPOCHS = 1
STEPS_PER_EPOCH = len(mix_up_set_sc) // BATCH_SIZE
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

AUTO = tf.data.AUTOTUNE
LEARNING_RATE = 0.03

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
class AdaMatch(keras.Model):
    def __init__(self, model, total_steps, tau=0.9):
        super(AdaMatch, self).__init__()
        self.model = model
        self.tau = tau  # Denotes the confidence threshold
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.total_steps = total_steps
        self.current_step = tf.Variable(0, dtype="int64")
        self.run_eagerly = True
    @property
    def metrics(self):
        return [self.loss_tracker]

    # This is a warmup schedule to update the weight of the
    # loss contributed by the target unlabeled samples. More
    # on this in the text.
    def compute_mu(self):
        pi = tf.constant(np.pi, dtype="float32")
        step = tf.cast(self.current_step, dtype="float32")
        return 0.5 - tf.cos(tf.math.minimum(pi, (2 * pi * step) / self.total_steps)) / 2

    def train_step(self, data):
        ## Unpack and organize the data ##
        source_ds, target_ds = data
        (source_w, source_labels), (source_s, _) = source_ds
        (
            (target_w, _),
            (target_s, _),
        ) = target_ds  # Notice that we are NOT using any labels here.

        combined_images = tf.concat([source_w, source_s, target_w, target_s], 0)
        combined_source = tf.concat([source_w, source_s], 0)

        total_source = tf.shape(combined_source)[0]
        total_target = tf.shape(tf.concat([target_w, target_s], 0))[0]

        with tf.GradientTape() as tape:
            ## Forward passes ##
            combined_logits = self.model(combined_images, training=True)
            z_d_prime_source = self.model(
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
            c_tau = self.tau * final_sum
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

            t = self.compute_mu()  # Compute weight for the target loss
            total_loss = source_loss + (t * target_loss)
            self.current_step.assign_add(
                1
            )  # Update current training step for the scheduler

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}
# %%
def wide_basic(x, n_input_plane, n_output_plane, stride):
    conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]

    n_bottleneck_plane = n_output_plane

    # Residual block
    for i, v in enumerate(conv_params):
        if i == 0:
            if n_input_plane != n_output_plane:
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                convs = x
            else:
                convs = layers.BatchNormalization()(x)
                convs = layers.Activation("relu")(convs)
            convs = layers.Conv2D(
                n_bottleneck_plane,
                (v[0], v[1]),
                strides=v[2],
                padding=v[3],
                kernel_initializer=INIT,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                use_bias=False,
            )(convs)
        else:
            convs = layers.BatchNormalization()(convs)
            convs = layers.Activation("relu")(convs)
            convs = layers.Conv2D(
                n_bottleneck_plane,
                (v[0], v[1]),
                strides=v[2],
                padding=v[3],
                kernel_initializer=INIT,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                use_bias=False,
            )(convs)

    # Shortcut connection: identity function or 1x1
    # convolutional
    #  (depends on difference between input & output shape - this
    #   corresponds to whether we are using the first block in
    #   each
    #   group; see `block_series()`).
    if n_input_plane != n_output_plane:
        shortcut = layers.Conv2D(
            n_output_plane,
            (1, 1),
            strides=stride,
            padding="same",
            kernel_initializer=INIT,
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            use_bias=False,
        )(x)
    else:
        shortcut = x

    return layers.Add()([convs, shortcut])


# Stacking residual units on the same stage
def block_series(x, n_input_plane, n_output_plane, count, stride):
    x = wide_basic(x, n_input_plane, n_output_plane, stride)
    for i in range(2, int(count + 1)):
        x = wide_basic(x, n_output_plane, n_output_plane, stride=1)
    return x


def get_network(image_size=32, num_classes=10):
    n = (DEPTH - 4) / 6
    n_stages = [16, 16 * WIDTH_MULT, 32 * WIDTH_MULT, 64 * WIDTH_MULT]

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = Rescaling(scale=1.0 / 255)(inputs)

    conv1 = layers.Conv2D(
        n_stages[0],
        (3, 3),
        strides=1,
        padding="same",
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        use_bias=False,
    )(x)

    ## Add wide residual blocks ##

    conv2 = block_series(
        conv1,
        n_input_plane=n_stages[0],
        n_output_plane=n_stages[1],
        count=n,
        stride=(1, 1),
    )  # Stage 1

    conv3 = block_series(
        conv2,
        n_input_plane=n_stages[1],
        n_output_plane=n_stages[2],
        count=n,
        stride=(2, 2),
    )  # Stage 2

    conv4 = block_series(
        conv3,
        n_input_plane=n_stages[2],
        n_output_plane=n_stages[3],
        count=n,
        stride=(2, 2),
    )  # Stage 3

    batch_norm = layers.BatchNormalization()(conv4)
    relu = layers.Activation("relu")(batch_norm)

    # Classifier
    trunk_outputs = layers.GlobalAveragePooling2D()(relu)
    outputs = layers.Dense(
        num_classes, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(trunk_outputs)

    return keras.Model(inputs, outputs)
# %%
wrn_model = get_network(image_size=224, num_classes=9)
print(f"Model has {wrn_model.count_params()/1e6} Million parameters.")
# %%
reduce_lr = keras.optimizers.schedules.CosineDecay(LEARNING_RATE, TOTAL_STEPS, 0.25)
optimizer = keras.optimizers.Adam(reduce_lr)

adamatch_trainer = AdaMatch(model=wrn_model, total_steps=TOTAL_STEPS)
adamatch_trainer.compile(optimizer=optimizer)
# %%
total_ds = tf.data.Dataset.zip((final_source_ds, final_target_ds))
adamatch_trainer.fit(total_ds, epochs=EPOCHS)

# %% Model load
adamatch_trained_model = adamatch_trainer.model
# adamatch_trained_model = keras.models.load_model("/nas001/users/jw.lee/adamatch/adamatch_trained_model.h5")
# %%
sound_dataset_generator_tg = dgs.eval_dataset_load(file_name_list_tg_test, file_label_list_tg_test, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, CHANNEL_NAME)
target_ds_w_test = tf.data.Dataset.from_generator(sound_dataset_generator_tg.eval_gen, (tf.float32, tf.float64), ((224, 224, 3), ()))
target_ds_w_test = target_ds_w_test.batch(BATCH_SIZE*3)


# %%
sound_dataset_generator_sc = dgs.eval_dataset_load(file_name_list_sc, file_label_list_sc, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, CHANNEL_NAME)
source_ds_w_test = tf.data.Dataset.from_generator(sound_dataset_generator_sc.eval_gen, (tf.float32, tf.float64), ((224, 224, 3), ()))
source_ds_w_test = source_ds_w_test.batch(BATCH_SIZE*3)


# %%
adamatch_trained_model.compile(metrics=keras.metrics.SparseCategoricalAccuracy())
_, accuracy = adamatch_trained_model.evaluate(target_ds_w_test)
print(f"Accuracy on source test set: {accuracy * 100:.2f}%")

# %%
prediction = adamatch_trained_model.predict(source_ds_w_test)

# %%
# Utility function for preprocessing the source test set.

adamatch_trained_model = adamatch_trainer.model
adamatch_trained_model.compile(metrics=keras.metrics.SparseCategoricalAccuracy())

def prepare_test_ds_source(image, label):
    image = tf.image.resize_with_pad(image, RESIZE_TO, RESIZE_TO)
    image = tf.tile(image, [1, 1, 3])
    return image, label


source_test_ds = tf.data.Dataset.from_tensor_slices((mnist_x_test, mnist_y_test))
source_test_ds = (
    source_test_ds.map(prepare_test_ds_source, num_parallel_calls=AUTO)
    .batch(TARGET_BATCH_SIZE)
    .prefetch(AUTO)
)

# Evaluation on the source test set.



# %%
# %%
mix_up_set_tg_test = data_utils.make_mixup_fileset(file_name_list_tg_test, category_list_tg, file_label_list_tg_test, seed=1004, mixup=False)
sound_dataset_generator_tg = dgs.sound_dataset_generator(mix_up_set_tg_test, file_name_list_tg_test, file_label_list_tg_test, SR, IMG_HEIGHT, IMG_WIDTH, TIME_SIZE, MIXUP_ALPHA, MIXUP_BETA, CHANNEL_NAME, is_train = False)
target_ds_w_test = tf.data.Dataset.from_generator(sound_dataset_generator_tg.weak_aug_ds, (tf.float32, tf.float64), ((224, 224, 3), (9)))
target_ds_w_test = target_ds_w_test.batch(BATCH_SIZE)

# %%
_, accuracy = adamatch_trained_model.evaluate(target_ds_w_test)
print(f"Accuracy on source test set: {accuracy * 100:.2f}%")
