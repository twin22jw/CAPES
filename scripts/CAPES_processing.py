#!/usr/bin/env python3.10
import os
from osgeo import gdal
import numpy as np
import scriptconfig as scfg
# from pathlib import Path

"""
Command line:

python3 CAPES_processing.py \
    --potentialCC_path='$HOME/CAPES/prediction/31UDQ/S6/31UDQ_2019_S8_test_00_prediction.tif' \
    --cv_path='$HOME/CAPES/sample_data/COLD_cv_blue.tif' 
    
"""
class CAPESprocessing(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    potentialCC_path = scfg.Value(None, type=str, help='filepath for potential construction change area (model prediction result')
    cv_path = scfg.Value(None, help='filepath for COLD change vector image to get break pixel information')
   

def main(cmdline=1, **kwargs):
    config = CAPESprocessing.cli(cmdline=cmdline, data=kwargs, strict=True)
    print(config)
    
    potentialCC_path = config['potentialCC_path']
    cv_path = config['cv_path']
    
    # Load data
    potentialCC_ds = gdal.Open(potentialCC_path)
    potentialCC_array = potentialCC_ds.GetRasterBand(1).ReadAsArray()
    out_img_info = get_img_info(potentialCC_path)

    cv_ds = gdal.Open(cv_path)
    cv_array = cv_ds.GetRasterBand(1).ReadAsArray()
    
    binary_array = np.where(cv_array != 0.5, 1, 0) # 1 for break pixel, 0 for no-change pixel
    CC_array = potentialCC_array * binary_array
    
    outname = potentialCC_path.replace('.tif', '_CAPES.tif')
    save_geotiff(outname, CC_array, out_img_info[0], out_img_info[1], out_img_info[3], out_img_info[4])
    print('saved:', outname)    

def get_img_info(input_img_path):
    img = gdal.Open(input_img_path)
    geotransform = img.GetGeoTransform()    
    proj = img.GetProjection()
    Numband = img.RasterCount
    img_arr = img.ReadAsArray().astype(np.float32)
    if Numband == 1:
        Xsize = img_arr.shape[1]
        Ysize = img_arr.shape[0]
    else:
        Xsize = img_arr.shape[2]
        Ysize = img_arr.shape[1]
    
    img_info = [Xsize, Ysize, Numband, geotransform, proj]
    return img_info


def save_geotiff(outname, np_array, Xsize, Ysize, geo_transform, projection):   
    driver = gdal.GetDriverByName('GTiff')    
    pred_ds = driver.Create(outname, Xsize, Ysize, 1, gdal.GDT_Int16)
    pred_ds.SetGeoTransform(geo_transform)
    pred_ds.SetProjection(projection)
    #pred_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    pred_ds.GetRasterBand(1).WriteArray(np_array)
    pred_ds = None
    
    
if __name__ == "__main__":
    main()