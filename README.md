# Deep-Learning-floods

# Creating an environment to run this tool    
We do not provide a dependency file but steps to create an environment to run our tool.
Create a new environment named tf_gpu and install tensorflow with gpu    
````conda create --name tf_gpu tensorflow-gpu   ````  
Activate environment  
````conda activate tf_gpu  ````  
Install gdal  
````conda install -c conda-forge gdal  ````  

Install earth engine and related packages  
````pip install earthengine-api geemap geedim  ````  

Install packages for geospatial data analysis and Ml  
````pip install pandas xarray rioxarray rasterio tqdm jupyter joblib catboost  ````
