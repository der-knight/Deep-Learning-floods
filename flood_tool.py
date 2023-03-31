import ee 
import geemap
import os
from osgeo import gdal
from joblib import Parallel,delayed
import rasterio
from rasterio.merge import merge
import xarray as xr
import pandas as pd 
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def Initialize():
    # Initialize the Earth Engine module.
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
    print('Earth Engine initialized.')

def gpu_check():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus)>0:
        print(len(gpus), "Physical GPUs,")
    else:
        print('GPU not found')

def emptyDirectory(dirPath, ext = "*", verbose = True):
    allFiles = glob.glob(dirPath + "/" + ext)
    if verbose:
        print(str(len(allFiles)) + " files found in " + dirPath + " --> Will be deleted.")
    for f in allFiles:
        os.remove(f)

def createDirectory(dirPath=os.path.expanduser('~/Downloads/SAR_flood'), emptyExistingFiles = False, verbose = True):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
        if verbose:
            print("Folder not found!!!   " + dirPath + " created.")
            print('------Folder Creation Done!!!--------')
    else:
        print('%s --> Folder exists!!!'%dirPath)
        print('------------------Using existing folder!!!-----------------')
        if emptyExistingFiles:
            emptyDirectory(dirPath, verbose = verbose)
    return(dirPath)

def download_gee_image(band,file_name,geometry):
    geemap.download_ee_image(band,file_name,crs="EPSG:4326",region=geometry, scale=10)

def get_files_from_gee(date,geometry,path):
    ee_date=ee.Date(date)
    ee_geometry =ee.Geometry.Rectangle(geometry)
    name='gee_files'+'_'.join([str(elem) for elem in geometry])
    file_path=createDirectory(os.path.join(path,name))

    dem=ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(ee_geometry)
    slope=ee.Terrain.slope(dem).clip(ee_geometry)
    jrc=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('seasonality').clip(ee_geometry)


    s1_file_name=os.path.join(file_path,'s1_'+'_'.join([str(elem) for elem in geometry])+'.tif')
    dem_file_name=os.path.join(file_path,'dem_'+'_'.join([str(elem) for elem in geometry])+'.tif')
    slope_file_name=os.path.join(file_path,'slope_'+'_'.join([str(elem) for elem in geometry])+'.tif')
    jrc_file_name=os.path.join(file_path,'jrc_'+'_'.join([str(elem) for elem in geometry])+'.tif')

    sen1= ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(ee_geometry).filterDate(ee_date,ee_date.advance(10,'days')).mosaic().clip(ee_geometry).select(['VV','VH'])
    list=[sen1,dem,slope,jrc]
    name=[s1_file_name,dem_file_name,slope_file_name,jrc_file_name]
    if geometry[2]-geometry[0]+geometry[3]-geometry[1]>0.5:
        y=input('The area you entered is greater than 0.5 degrees.Press Y if you want to continue')
        if str(y)=='Y':
            print("--------------Files download getting ready ---------------------")
            Parallel(n_jobs=5,backend='threading')(delayed(download_gee_image)(list[i],name[i],ee_geometry) for i in range(len(list)))
        else:
            print('Re-enter new coordinates')
    else:
        print("--------------Files download getting ready ---------------------")
        Parallel(n_jobs=5,backend='threading')(delayed(download_gee_image)(list[i],name[i],ee_geometry) for i in range(len(list)))
        print('--------------Files Download Done!!!------------------------')
    return(s1_file_name,dem_file_name,slope_file_name,jrc_file_name)

def chunk_band(path,input_filename ,tile_size=512):
        name=input_filename.split('\\')[-1].split('.')
        name='.'.join([str(elem) for elem in name[:-1]])
        output_file=createDirectory(os.path.join(path,name))
        output_filename = output_file+'/tile_'
        ds = gdal.Open(input_filename)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        x_grid= int(xsize/tile_size)
        y_grid= int(ysize/tile_size)
        for i in range(0, x_grid):
            for j in range(0, y_grid):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i*tile_size)+ ", " + str(j*tile_size) + ", " + str(tile_size) + ", " + str(tile_size) + " "  + str(input_filename) + " " + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                os.system(com_string)
        # edge case
        for i in range(x_grid):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i*tile_size)+ ", " + str(ysize-tile_size) + ", " + str(tile_size) + ", " + str(tile_size) + " "  + str(input_filename) + " " + str(output_filename)+"edg_y" + str(i) +  ".tif"
                os.system(com_string)
        for j in range(y_grid):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(xsize-tile_size)+ ", " + str(j*tile_size) + ", " + str(tile_size) + ", " + str(tile_size) + " "  + str(input_filename) + " " + str(output_filename)+"edg_x" + str(j) + ".tif"
                os.system(com_string)
        com_string = "gdal_translate -of GTIFF -srcwin " + str(xsize-tile_size)+ ", " + str(ysize-tile_size) + ", " + str(tile_size) + ", " + str(tile_size) + " "  + str(input_filename) + " " + str(output_filename)+"true_edge" + ".tif"
        os.system(com_string)
        
        print('------Files Chunking Done!!!--------')
        print(f'Files are saved in {output_file}')
        return(output_file)
def dimension_check(df):    
    files=[]
    for column in df.columns:
        for file in df[column]:
            if len(xr.open_rasterio(file).x)!=512:
                files.append(file)
    if len(files)==0:
        print('Files checked for dimensions, good to go')
    else:
        print('Problem in dimensions, all files are not 512*512',files)



def get_location_database(jrc_chunked,dem_chunked,slope_chunked,s1_chunked):
    files_location=pd.DataFrame()
    files_location['jrc_data']=glob.glob(os.path.join(jrc_chunked,'*.tif'))
    files_location['dem_data']=glob.glob(os.path.join(dem_chunked,'*.tif'))
    files_location['slope_data']=glob.glob(os.path.join(slope_chunked,'*.tif'))
    files_location['s1_data']=glob.glob(os.path.join(s1_chunked,'*.tif'))
    print('------Files Location Database Created successfully!!!--------')
    return(files_location)



def get_images(df,i,probability_path, individual_probability):
    features = []                                          
    #load labels

    cols = ["s1_data", "dem_data","slope_data", "jrc_data"]

    images = []

    for col in cols:
        with rasterio.open(df.iloc[i][col]) as img:
            name=df.iloc[i][col].split('.')[0]
            #load the tif file
            if col == "s1_data":
                ar=np.float32(np.clip(img.read(1), -35, 0)/-35)
                ar[np.isnan(ar)] = 0
                images.append(ar)
                ar=np.float32(np.clip(img.read(2), -42, -5)/-42)
                ar[np.isnan(ar)] = 0
                images.append(ar)
                dict=img.meta
                name='probability_'+df.iloc[i]['s1_data'].split(os.sep)[-1].split('.')[0]
                name=os.path.join(probability_path,name+'.tif')
                if individual_probability:
                    new_dataset = rasterio.open(
                            name, "w", 
                            driver = dict['driver'],
                            height = dict['height'],
                            width = dict['width'],
                            count = 5,
                            nodata =dict['nodata'],
                            dtype = dict['dtype'],
                            crs = dict['crs'],
                            transform = dict['transform'])
                else:
                        
                        new_dataset = rasterio.open(
                        name, "w", 
                        driver = dict['driver'],
                        height = dict['height'],
                        width = dict['width'],
                        count = 2,
                        nodata =dict['nodata'],
                        dtype = dict['dtype'],
                        crs = dict['crs'],
                        transform = dict['transform'])
    

            elif col == "dem_data":
                ar=img.read(1)
                #clip values > 255 and converto to uint8
                ar[ar==-32768]=0
                ar=np.float32(ar/np.max(ar))
                images.append(ar)
            elif col=="slope_data":
                ar=np.float32(img.read(1)/np.max(img.read(1)))
                ar[np.isnan(ar)] = 0
                images.append(ar)

            elif col=="jrc_data":
                img=np.float32(img.read(1))
                img[img==-128]=0
                img[img>0]=1
                images.append(img)
    features.append(np.stack(images, axis=-1)) 
    dl_files=np.array(features)
    return dl_files,new_dataset

def create_combined_probability_tiff(probability_path):
    file=glob.glob(os.path.join(probability_path,'*.tif'))
    src_files_to_mosaic = []
    for fp in file:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy() 

    out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans})
    file_name=os.path.join(os.path.dirname(probability_path),probability_path.split('\\')[-1]+'combined.tif')
    with rasterio.open(file_name, "w", **out_meta) as dest:
        dest.write(mosaic)

def get_probability(model,files):
    probability=[]
    for i in range(3):
        probability.append(model[i].predict(files,verbose=0))
    return probability


def predict_classes(probability_array):
    for i in range(len(probability_array)):
        if (probability_array[i]>0.5):
            probability_array[i]=1
        else:
            probability_array[i]=0
    return (probability_array) 

def slope_postprocessing(location_database,ensemble_class,i):
    slope_mask=(rasterio.open(location_database.iloc[i]['slope_data']).read(1)<5).astype('float32').ravel()
    ensemble_class=slope_mask*ensemble_class
    return ensemble_class

def create_output_tiff(models,location_database,probability_path,individual_probability=False):
    for i in tqdm(range(len(location_database))):
        files,tiff_file=get_images(location_database,i,probability_path,individual_probability)
        probability=get_probability(models,files)
        probability_sum=np.array(probability[0]+probability[1]+probability[2]).ravel()
        ensemble_class=predict_classes(probability_sum/3)
        ensemble_class=slope_postprocessing(location_database,ensemble_class,i)

        uncertainty=np.zeros(len(ensemble_class))
        for i in range(len(ensemble_class)):
            if ensemble_class[i]==1:
                uncertainty[i]=2-(2/3*(probability_sum[i]))
        else:
            uncertainty[i]=2/3*(probability_sum[i])
        ensemble_class=ensemble_class.reshape(512,512)
        uncertainty=uncertainty.reshape(512,512)

        if individual_probability:
            probability[0]=probability[0].reshape(512,512)
            probability[1]=probability[1].reshape(512,512)
            probability[2]=probability[2].reshape(512,512)
            tiff_file.write(probability[0], 1)
            tiff_file.set_band_description(1, 'model1')
            tiff_file.write(probability[1], 2)
            tiff_file.set_band_description(2, 'model2')
            tiff_file.write(probability[2], 3)
            tiff_file.set_band_description(3, 'model3')
            tiff_file.write(ensemble_class, 4)
            tiff_file.set_band_description(4, 'flood_class')
            tiff_file.write(uncertainty, 5)
            tiff_file.set_band_description(5, 'uncertainty')
        else:
            tiff_file.write(ensemble_class, 1)
            tiff_file.set_band_description(1, 'flood_class')
            tiff_file.write(uncertainty, 2)
            tiff_file.set_band_description(2, 'uncertainty')
        tiff_file.close()

def split_raster_bands(probability_path):
    name=probability_path.split('\\')[-1]
    raster_path=os.path.join(os.path.dirname(probability_path),name+'combined'+'.tif')
    with rasterio.open(raster_path) as src:
        count=rasterio.open(raster_path).count
        if count==5:
            file_names=['model1','model2','model3','ensemble_flood','uncertainty']
            out_meta = src.meta.copy() 
            out_meta['count']=1
            for i, band in enumerate(src.read()):
                output_path = os.path.join(os.path.dirname(probability_path),file_names[i]+name[11:]+".tif")
                with rasterio.open(output_path, "w",**out_meta) as dst:
                    dst.write(band, 1)
        else:
            file_names=['ensemble_flood','uncertainty']
            out_meta = src.meta.copy() 
            out_meta['count']=1
            for i, band in enumerate(src.read()):
                output_path = os.path.join(os.path.dirname(probability_path),file_names[i]+name[11:]+".tif")
                with rasterio.open(output_path, "w",**out_meta) as dst:
                    dst.write(band, 1)

def remove_data(path):
    folders=[x[0] for x in os.walk(path)]
    for folder in folders[1:]:   
        emptyDirectory(folder)
        os.rmdir(folder)
    os.remove(glob.glob(os.path.join(path,'probability*.tif'))[0])



def display_input_chip(location_database,rand=True):
    if rand:
        i=np.random.randint(0,len(location_database)-1)
    else:
        i=input('Enter a random number between {} and {}: '.format(0,len(location_database)-1))
    i=int(i)
    
    cm = LinearSegmentedColormap.from_list( 'cm', ['#000000','#FFFFFF','#0000FF'], N=3)
    water_cm= LinearSegmentedColormap.from_list( 'cm', ['#FFFFFF','#0000FF'], N=2)

    file=xr.open_rasterio(location_database.iloc[i]['s1_data'])
    jrc=xr.open_rasterio(location_database.iloc[i]['jrc_data'])
    dem=xr.open_rasterio(location_database.iloc[i]['dem_data'])
    slope=xr.open_rasterio(location_database.iloc[i]['slope_data'])

    fig,ax=plt.subplots(1,5,figsize=(24,8),)
    num_colors = 20
    cmap = plt.get_cmap('Greys_r', num_colors)

    im=ax[0].imshow(file[0],cmap=cmap,vmin=-25,vmax=-5)
    ax[0].set_title('SAR Band - VV')
    divider = make_axes_locatable(ax[0])
    cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
    # Create colorbar
    ax[0].figure.add_axes(cax)
    cbar = ax[0].figure.colorbar(im, cax = cax,orientation = 'horizontal')
    ax[0].set_xticks([])
    ax[0].set_yticks([])


    im=ax[1].imshow(file[1],cmap=cmap,vmin=-35,vmax=0)
    ax[1].set_title('SAR band-VH')
    divider = make_axes_locatable(ax[1])
    cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
    # Create colorbar
    ax[1].figure.add_axes(cax)
    cbar = ax[1].figure.colorbar(im, cax,orientation = 'horizontal')
    ax[1].set_yticks([])
    ax[1].set_xticks([])


    im=ax[2].imshow(dem[0],cmap='terrain')
    ax[2].set_title('DEM')
    divider = make_axes_locatable(ax[2])
    cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
    # Create colorbar
    ax[2].figure.add_axes(cax)
    cbar = ax[2].figure.colorbar(im, cax,orientation = 'horizontal')
    ax[2].set_yticks([])
    ax[2].set_xticks([])

    im=ax[3].imshow(slope[0])
    ax[3].set_title('Slope')
    divider = make_axes_locatable(ax[3])
    cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
    # Create colorbar
    ax[3].figure.add_axes(cax)
    cbar = ax[3].figure.colorbar(im, cax,orientation = 'horizontal')
    ax[3].set_yticks([])
    ax[3].set_xticks([])


    im=ax[4].imshow(jrc[0],cmap=water_cm)
    divider = make_axes_locatable(ax[4])
    cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
    ax[4].set_title('JRC Permanent Water')
    legend_labels ={'JRC Water Layer':'#0000FF'}
    patches = [Patch(color=color, label=label)for label, color in legend_labels.items()]
    ax[4].legend(handles=patches,bbox_to_anchor=(0.8, -0.00), prop={'size':14},title_fontsize= 14,facecolor='white')
    ax[4].set_yticks([])
    ax[4].set_xticks([])
    
    plt.suptitle(f'Input Datasets for chip {i}', fontsize=40)
    plt.grid(False)
    fig.tight_layout(rect=[0, 0.01, 1, 1.2])
    plt.show()




def display_output_chip(probability_path,rand=True):
    files=glob.glob(probability_path+'/*.tif')
    if rand:
        i=np.random.randint(0,len(files)-1)
    else:
        i=input('Enter a random number between {} and {}: '.format(0,len(files)-1))
    i=int(i)
    image=xr.open_rasterio(files[i])
    length=len(image.band)
    water_cm= LinearSegmentedColormap.from_list( 'cm', ['#FFFFFF','#0000FF'], N=2)
    uncertainty_cm= LinearSegmentedColormap.from_list( 'cm', ['#FFFFFF','#FF0000'], N=10)
    if length==2:
        im=image[0]
        fig,ax=plt.subplots(1,2,figsize=(28,16))
        ax[0].imshow(im,vmin=0,vmax=1,cmap=water_cm)
        divider = make_axes_locatable(ax[0])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        ax[0].set_title('Ensemble Flood layer',fontsize=20)
        legend_labels ={'Water':'#0000FF'}
        patches = [Patch(color=color, label=label)for label, color in legend_labels.items()]
        ax[0].legend(handles=patches,bbox_to_anchor=(0.6, 0), prop={'size':20},title_fontsize= 20,facecolor='white')
        ax[0].set_yticks([])
        ax[0].set_xticks([])

        im=image[1]
        im=ax[1].imshow(im,cmap=uncertainty_cm,vmin=0,vmax=1)
        ax[1].set_title('Uncertainty',fontsize=20)
        divider = make_axes_locatable(ax[1])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # Create colorbar
        ax[1].figure.add_axes(cax)
        cbar = ax[1].figure.colorbar(im, cax,orientation = 'horizontal')
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        plt.suptitle(f'Outputs for chip {i}', fontsize=40)
        plt.grid(False)
        fig.tight_layout()
        plt.show()
    else:
        fig,ax=plt.subplots(1,5,figsize=(24,8))

        for j in range(4):
            im=image[j]
            ax[j].imshow(im,vmin=0,vmax=1,cmap=water_cm)
            divider = make_axes_locatable(ax[j])
            cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
            legend_labels ={'Water':'#0000FF'}
            patches = [Patch(color=color, label=label)for label, color in legend_labels.items()]
            ax[j].legend(handles=patches,bbox_to_anchor=(0.6, 0), prop={'size':10},title_fontsize= 10,facecolor='white')
            ax[j].set_yticks([])
            ax[j].set_xticks([])
        
        im=image[4]
        im=ax[4].imshow(im,cmap=uncertainty_cm,vmin=0,vmax=1)
        ax[4].set_title('Uncertainty',fontsize=10)
        divider = make_axes_locatable(ax[4])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # Create colorbar
        ax[4].figure.add_axes(cax)
        cbar = ax[4].figure.colorbar(im, cax,orientation = 'horizontal')
        ax[4].set_yticks([])
        ax[4].set_xticks([])

        ax[0].set_title('Model 1 flood layer',fontsize=10)
        ax[1].set_title('Model 2 flood layer',fontsize=10)
        ax[2].set_title('Model 3 flood layer',fontsize=10)
        ax[3].set_title('Ensemble flood layer',fontsize=10)
        ax[4].set_title('Uncertainty',fontsize=10)
        plt.suptitle(f'Outputs for chip {i}', fontsize=40)
        plt.grid(False)
        fig.tight_layout(rect=[0, 0.01, 1, 1.2])
        plt.show()
