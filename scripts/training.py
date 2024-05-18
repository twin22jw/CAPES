#!/usr/bin/env python3.10
import scriptconfig as scfg
from pathlib import Path
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm
from tensorflow import pad
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

"""
Command line example:

python3 training.py \
    --rootpath=$HOME/Document \
    --scenario='S6' \
    --HLS_tile='31UDQ' \
    --target_img_num=1000

"""

class Training(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    rootpath = scfg.Value(None, type=str, help='root path')
    scenario = scfg.Value(True, help='Lansat anomaly period, default is True')
    HLS_tile = scfg.Value(None, help='HLS tile, such as 31UDQ')
    method_type = scfg.Value('CAPES', choices=["CAPES", "ECTDC"], help='Method for extracting training data, choice: CAPES, ECTDC, default is CAPES')
    target_img_num = scfg.Value(9500, help='Number of images to be generated, default is 9500')
    batch_size = scfg.Value(32, help='batch size, default is 32')
    epochs = scfg.Value(100, help='epochs, default is 100')
    
def main(cmdline=1, **kwargs):
    config = Training.cli(cmdline=cmdline, data=kwargs, strict=True)
    print(config)

    rootpath = Path(config['rootpath'])
    scenario = config['scenario']
    HLS_tile = config['HLS_tile']
    method_type = config['method_type']
    target_img_num = config['target_img_num']
    batch_size = config['batch_size']
    epochs = config['epochs']
    
    aug_folder = rootpath / method_type / 'sample_data' / HLS_tile / scenario / 'train' / 'aug'
    
    # Load images and label
    Images_hf  = h5py.File(aug_folder / f'aug_{HLS_tile}_{scenario}_images_{target_img_num}.h5', 'r')
    images = np.array(Images_hf.get('Images'))
    print(images.shape)
    
    Labels_hf  = h5py.File(aug_folder / f'aug_{HLS_tile}_{scenario}_labels_{target_img_num}.h5', 'r')
    labels = np.array(Labels_hf.get('Labels'))
    
    labels = to_categorical(labels, num_classes = 2)
    #labels = np_utils.to_categorical(labels, num_classes = 2) # tensorflow 2.x
    labels = labels.reshape((images.shape[0], images.shape[1], images.shape[1], 2))    

    Img_Width    = images.shape[1]
    Img_Height   = images.shape[2]
    Img_Channels = images.shape[3]

    inputs  = Input((Img_Width, Img_Height, Img_Channels))
    
    input_pad = ReflectionPadding2D(padding=(1, 1))(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(input_pad)
    c1 = BatchNormalization(axis=-1)(c1)
    c1 = Dropout(0.1)(c1)
    c1 = ReflectionPadding2D(padding=(1, 1))(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c1)
    c1 = BatchNormalization(axis=-1)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = ReflectionPadding2D(padding=(1, 1))(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c2)
    c2 = BatchNormalization(axis=-1)(c2)
    c2 = Dropout(0.1)(c2)
    c2 = ReflectionPadding2D(padding=(1, 1))(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c2)
    c2 = BatchNormalization(axis=-1)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = ReflectionPadding2D(padding=(1, 1))(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c3)
    c3 = BatchNormalization(axis=-1)(c3)
    c3 = Dropout(0.2)(c3)
    c3 = ReflectionPadding2D(padding=(1, 1))(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c3)
    c3 = BatchNormalization(axis=-1)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = ReflectionPadding2D(padding=(1, 1))(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c4)
    c4 = BatchNormalization(axis=-1)(c4)
    c4 = Dropout(0.2)(c4)
    c4 = ReflectionPadding2D(padding=(1, 1))(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c4)
    c4 = BatchNormalization(axis=-1)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = ReflectionPadding2D(padding=(1, 1))(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c5)
    c5 = BatchNormalization(axis=-1)(c5)
    c5 = Dropout(0.3)(c5)
    c5 = ReflectionPadding2D(padding=(1, 1))(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c5)
    c5 = BatchNormalization(axis=-1)(c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid')(c5)
    u6 = concatenate([u6, c4])

    c6 = ReflectionPadding2D(padding=(1, 1))(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c6)
    c6 = Dropout(0.2)(c6)
    c6 = ReflectionPadding2D(padding=(1, 1))(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid')(c6)
    u7 = concatenate([u7, c3])

    c7 = ReflectionPadding2D(padding=(1, 1))(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c7)
    c7 = Dropout(0.2)(c7)
    c7 = ReflectionPadding2D(padding=(1, 1))(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(c7)
    u8 = concatenate([u8, c2])

    c8 = ReflectionPadding2D(padding=(1, 1))(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c8)
    c8 = Dropout(0.1)(c8)
    c8 = ReflectionPadding2D(padding=(1, 1))(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid')(c8)
    u9 = concatenate([u9, c1], axis=3)

    c9 = ReflectionPadding2D(padding=(1, 1))(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c9)
    c9 = Dropout(0.1)(c9)
    c9 = ReflectionPadding2D(padding=(1, 1))(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c9)

    outputs = Conv2D(2, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [f1])
    
    checkpointer = ModelCheckpoint(rootpath / method_type / 'trained_models' / HLS_tile / scenario / f'{method_type}_{HLS_tile}_{scenario}_epoch{{epoch:02d}}_val_loss_{{val_loss:.4f}}.h5', verbose = 1)

    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
        epsilon=0.0001, cooldown=4, min_lr=10e-7)
    
    results = model.fit(images, labels, validation_split = 0.10, batch_size = batch_size, epochs= epochs, callbacks = [checkpointer, LR])
    
        
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = padding
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return K.mean(f1)
    
if __name__ == "__main__":
    main()


