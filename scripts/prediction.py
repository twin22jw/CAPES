#!/usr/bin/env python3.10
import os
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import scriptconfig as scfg
from pathlib import Path
import tensorflow as tf
from tensorflow import pad
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from keras import backend as K

"""
Command line:

python3 prediction.py \
    --rootpath=$HOME/Document \
    --scenario='S6' \
    --HLS_tile='31UDQ'    

"""

class Prediction(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    rootpath = scfg.Value(None, type=str, help='root path')
    scenario = scfg.Value(None, help='input scenario, S1 ~ S7')
    HLS_tile = scfg.Value(None, help='HLS tile, such as 52SDG')
    method_type = scfg.Value('CAPES', choices=["CAPES", "ECTDC"], help='Method for extracting training data, choice: CAPES, ECTDC, default is CAPES')
    model_path = scfg.Value(None, type=str, help='.h5 path of the trained model')
    class_num = scfg.Value(2, help='Number of class, default is 2')
   

def main(cmdline=1, **kwargs):
    config = Prediction.cli(cmdline=cmdline, data=kwargs, strict=True)
    print(config)
    
    rootpath = Path(config['rootpath'])
    scenario = config['scenario']
    HLS_tile = config['HLS_tile']
    method_type = config['method_type']
    model_path = config['model_path']
    class_num = config['class_num'] #number_of_classes
        
    input_test_dir = rootpath / method_type / 'sample_data' / HLS_tile / scenario / 'test'

    if model_path == None:
        model_path_list = [rootpath / method_type / 'trained_models' / model_pth for model_pth in os.listdir(rootpath / method_type / 'trained_models') if method_type in model_pth and HLS_tile in model_pth and scenario in model_pth]
        if not len(model_path_list) == 0:
            model_path = model_path_list[0]
        else:
            raise FileNotFoundError('No model found in the directory')
    
    if isinstance(model_path, Path):
        model_path = str(model_path)
    loss = model_path.split('.')[1]
    output_pred_dir = rootpath / method_type / 'prediction' / HLS_tile / scenario
    os.makedirs(output_pred_dir, exist_ok=True)
    
    evaluate(input_test_dir, model_path, output_pred_dir, HLS_tile, scenario, loss, class_num)


def save_image(image_data, path, geo_transform, projection):
    
    driver = gdal.GetDriverByName('GTiff')
    
    # Set Info of Image
    height, width = image_data.shape
    dataset       = driver.Create(path, width, height, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data)    
    

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding,            
        })
        return config
    
    
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def eval_image(input_image_path, model, output_image_path, number_of_classes):
    
    input_dataset = gdal.Open(input_image_path)
    input_image   = input_dataset.ReadAsArray().astype(np.float32)
    input_image = np.rollaxis(input_image, 0, 3)
    h, w, n     = input_image.shape    
    
    model_input_height, model_input_width, model_input_channels    = model.layers[0].input_shape[0][1:4]
    model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[1:4]

    padding_y = int((model_input_height - model_output_height)/2)
    padding_x = int((model_input_width - model_output_width)/2)
    assert model_output_channels == number_of_classes

    pred_lc_image = np.zeros((h, w, number_of_classes))
    mask = np.ones((h, w))

    irows, icols = [],[]
    batch_size   = 16
    minibatch    = []
    ibatch       = 0
    mb_array     = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))

    n_rows = int(h / model_output_height)
    n_cols = int(w / model_output_width)
    
    for row_idx in tqdm(range(n_rows)):
        for col_idx in range(n_cols):
            
            subimage = input_image[row_idx*model_output_height:row_idx*model_output_height + model_input_height,
                                   col_idx*model_output_width:col_idx*model_output_width + model_input_width, :] #/ 255.0
            
            if(subimage.shape == model.layers[0].input_shape[0][1:4]):
                
                mb_array[ibatch] = subimage
                ibatch += 1
                irows.append((row_idx*model_output_height + padding_y,row_idx*model_output_height + model_input_height - padding_y))
                icols.append((col_idx*model_output_width +  padding_x,col_idx*model_output_width  + model_input_width  - padding_x))

                if (ibatch) == batch_size:
                    
                    outputs = model.predict(mb_array)
                    for i in range(batch_size):
                        r0,r1 = irows[i]
                        c0,c1 = icols[i]

                        pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
                        mask[r0:r1, c0:c1] = 0

                    ibatch = 0
                    irows,icols = [],[]

    if ibatch > 0:
        outputs = model.predict(mb_array)
        for i in range(ibatch):
            r0,r1 = irows[i]
            c0,c1 = icols[i]

            pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
            mask[r0:r1, c0:c1] = 0


    label_image = np.ma.array(pred_lc_image.argmax(axis=-1), mask = mask)
    save_image(label_image.filled(255), output_image_path, input_dataset.GetGeoTransform(), input_dataset.GetProjection())

def evaluate(input_dir, model_path, output_dir, HLS_tile, scenario, loss, number_of_classes):
    model = load_model(model_path, 
                       custom_objects = {'ReflectionPadding2D':ReflectionPadding2D,  'f1':f1}) 
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".tif"):    
                pth     = os.path.join(root,f)
                out_pth = os.path.join(output_dir,f.split('.')[0] + '_prediction.tif')                     
                eval_image(pth, model, out_pth, number_of_classes)
                print('saved result to ' + out_pth)
                
                
if __name__ == "__main__":
    main()