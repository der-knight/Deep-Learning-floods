# Deep-Learning-floods
![image](https://github.com/der-knight/Deep-Learning-floods/blob/main/Images/Methodology%20flood.jpg)
# Creating an environment to run this tool    
We do not provide a dependency file but steps to create an environment to run our tool.   
Model Weights can be downloaded from  
https://csciitd-my.sharepoint.com/personal/cez208302_iitd_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcez208302%5Fiitd%5Fac%5Fin%2FDocuments%2FDeepSARFLOOD&ga=1  
clone this repository   
```` git clone https://github.com/der-knight/Deep-Learning-floods.git````    
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
