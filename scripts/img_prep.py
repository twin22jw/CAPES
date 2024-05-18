#!/usr/bin/env python3.10
import os, h5py, random
from osgeo import gdal
import numpy as np
import tensorflow as tf
import scriptconfig as scfg
from pathlib import Path
import gc

"""
Command line example:

python3 img_prep.py \
    --rootpath=$HOME/Document \
    --scenario='S6' \
    --HLS_tile='31UDQ' \

"""

class ImgPrepConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    rootpath = scfg.Value(None, type=str, help='root path')
    scenario = scfg.Value(None, help='input scenario, S1 ~ S7')
    HLS_tile = scfg.Value(None, help='HLS tile, such as 31UDQ')
    method_type = scfg.Value('CAPES', choices=["CAPES", "ECTDC"], help='Method for extracting training data, choice: CAPES, ECTDC, default is CAPES')
    target_img_num = scfg.Value(9500, help='Number of images to be generated, default is 9500')

    
def main(cmdline=1, **kwargs):
    config = ImgPrepConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    config.cli(data=kwargs)
    print(config)
    
    rootpath = Path(config['rootpath'])
    scenario = config['scenario']
    HLS_tile = config['HLS_tile']
    method_type = config['method_type']
    target_img_num = config['target_img_num']
    img_folder = rootpath / method_type / 'sample_data' / HLS_tile / scenario / 'train' / 'img'
    aug_folder = rootpath / method_type / 'sample_data' / HLS_tile / scenario / 'train' / 'aug'
    os.makedirs(aug_folder, exist_ok=True)
    
    random.seed(42)
    
    channel_by_scenario = {'S1': 6,
                           'S2': 12,
                           'S3': 18,
                           'S4': 48,
                           'S5': 54,
                           'S6': 60,
                           'S7': 66}
    
    if HLS_tile == '10SEG':
        CC_aug_f = 24
        B_aug_f = 2
    elif HLS_tile == '17SMS':
        CC_aug_f = 8
        B_aug_f = 2 
    elif HLS_tile == '18STJ':
        CC_aug_f = 9 
        B_aug_f = 3 
    elif HLS_tile == '21HUB':
        CC_aug_f = 18 
        B_aug_f = 5 
    elif HLS_tile == '31UDQ':
        CC_aug_f = 8
        B_aug_f = 3
    elif HLS_tile == '32TMR':
        CC_aug_f = 8 
        B_aug_f = 2
    elif HLS_tile == '36RUU':
        CC_aug_f = 9
        B_aug_f = 2
    elif HLS_tile == '49QGF':
        CC_aug_f = 9
        B_aug_f = 5
    elif HLS_tile == '52SDG':
        CC_aug_f = 10
        B_aug_f = 3
        
    Background = []
    Back = 0
    Construction = 0
    CC = []

    # Append Background list defined by over 99% of the case unique value is 0
    for items in os.listdir((os.path.join(img_folder, 'labels'))):
        if items.endswith(".jpg") or items.endswith(".tif"):

            output_image_path = os.path.join(os.path.join(img_folder, 'labels'), items)                      
            #print(output_image_path)
            output_image      = gdal.Open(output_image_path)
            output_image_band = output_image.GetRasterBand(1)                        
            label_image    = np.array(output_image.GetRasterBand(1).ReadAsArray())
            unique, counts = np.unique(label_image, return_counts = True)
            freq           = np.asarray((unique, counts)).T
            label_image[label_image == 3] = 0
            if(freq[np.where(unique == 0)[0][0], 1] > 256 * 256 * 95 / 100):    
                Back = Back + 1
            
                input_image_path = output_image_path.replace('labels', 'images')                
                input_image = gdal.Open(input_image_path)
                input_image = np.array(input_image.ReadAsArray())
                # Add New Dim to Label
                label_image = np.expand_dims(label_image, axis = 0)
                        
                # Concatenate Image and Mask
                merge = np.concatenate((input_image, label_image), axis = 0)
                        
                # Add Image to List                        
                if(0 in unique):
                    Background.append(merge) 
                        
    for items in os.listdir((os.path.join(img_folder, 'labels'))):
        if items.endswith(".tif"):

            output_image_path = os.path.join(os.path.join(img_folder, 'labels'), items)
            output_image      = gdal.Open(output_image_path)
            output_image_band = output_image.GetRasterBand(1)            
            
            label_image    = np.array(output_image.GetRasterBand(1).ReadAsArray())
            unique, counts = np.unique(label_image, return_counts = True)
            freq           = np.asarray((unique, counts)).T
            label_image[label_image == 3] = 0    
            # Check Image is dark or not
                            
            if(freq[np.where(unique == 0)[0][0], 1] > 256 * 256 / 4):
                Construction = Construction + 1
                input_image_path = output_image_path.replace('labels', 'images')
                input_image = gdal.Open(input_image_path)
                input_image = np.array(input_image.ReadAsArray())
                        
                # Add New Dim to Label
                label_image = np.expand_dims(label_image, axis = 0)
                        
                # Concatenate Image and Mask
                merge = np.concatenate((input_image, label_image), axis = 0)
                        
                # Add Image to List                        
                if(1 in unique):
                    CC.append(merge)

    Background_Show = np.array(Background)
    CC_Show = np.array(CC)
    channel = CC_Show.shape[1] - 1
    
    assert channel_by_scenario[scenario] == channel, f'channel mismatch: {channel_by_scenario[scenario]} != {channel}, check input scenario again!'

    Background_Show = np.rollaxis(Background_Show, 1, 4)
    CC_Show = np.rollaxis(CC_Show, 1, 4)
    Background_Image = Background_Show[:, :, :, 0:channel]
    Background_Label = Background_Show[:, :, :, channel]
    Background_Label = np.expand_dims(Background_Label, axis = -1)
    
    CC_Image = CC_Show[:, :, :, 0:channel]
    CC_Label = CC_Show[:, :, :, channel]
    CC_Label = np.expand_dims(CC_Label, axis = -1)

    Background = np.concatenate((Background_Image, Background_Label), axis = 3)
    CC = np.concatenate((CC_Image, CC_Label), axis = 3)
    
    augmented_CC = augment_data(CC, augementation_factor = CC_aug_f)
    augmented_B = augment_data(Background, augementation_factor = B_aug_f)
    data = np.vstack([augmented_CC, augmented_B])
    
    del augmented_CC, augmented_B
    gc.collect()
    
    if data.shape[0] < target_img_num:
        raise Exception(f'the number of image chip is less than {target_img_num}')
    
    np.random.shuffle(data)
    print("data shape", data.shape)
    images = data[:target_img_num, :, :, 0:channel]
    labels = data[:target_img_num, :, :, channel]
    labels = labels.astype(int)
    
    del data
    gc.collect()
    
    Images_hf  = h5py.File(aug_folder / f'aug_{HLS_tile}_{scenario}_images_{target_img_num}.h5', 'w')
    print("images shape", images.shape)
    Images_hf.create_dataset('Images', data = images)
    Images_hf.close()

    Labels_hf  = h5py.File(aug_folder / f'aug_{HLS_tile}_{scenario}_labels_{target_img_num}.h5', 'w')
    print("labels shape", labels.shape)
    Labels_hf.create_dataset('Labels', data = labels)
    Labels_hf.close()
    
    gc.collect()


def augment_data(dataset, augementation_factor=1, use_random_rotation=True, 
                 use_random_shear=True, use_random_shift=True, use_random_zoom=True):
    augmented_image        = []
    
    for num in range (0, dataset.shape[0]):
        for i in range(0, augementation_factor):
            
            # original image:
            augmented_image.append(dataset[num])
            
            # rotation
            rotation = [-180, -90, 90, 25, 72, 167, 245, 45, -45, 135, -135, 30, -30, 140, -140]
            if use_random_rotation:
                augmented_image.append(tf.keras.preprocessing.image.random_rotation(dataset[num], random.choice(rotation), row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect'))            

    return np.array(augmented_image)


if __name__ == '__main__':
    main()
