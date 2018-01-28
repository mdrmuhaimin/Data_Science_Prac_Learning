from pyspark.sql import SparkSession 
from pyspark.sql.types import *

import elevation_grid as eg
import numpy as np
import weather_predict as wp

import time
from datetime import date


sc = SparkSession.builder.getOrCreate()

train_data, test_data = wp.get_data('tmax-2').randomSplit([0.75, 0.25])
# train_data.show()
model = wp.get_best_weather_model(train_data)
print("Best parameters on test data:\n", wp.get_best_tvs_model_params(model))

el = eg.get_elevation(50.0, -123.0)
print("A place near Whistler, BC is {} m above sea level".format(el))

import matplotlib.pyplot as plt

lats, lons = np.meshgrid(np.arange(-90,90,1.0),np.arange(-180,180,1.0))
coords = np.array([np.array([late,lone]).T for late,lone in zip(lats,lons)])
coords = coords.reshape((coords.shape[0] * coords.shape[1], 2))
datas = [( date.today(), float(lat), float(lon), float(eg.get_elevation(lat, lon)) ) for lat, lon in coords]

schema = StructType([
    StructField('date', DateType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False)
])

datas = sc.createDataFrame(datas, schema=schema)
data_pred = model.transform(datas).drop("features")
data_pred.show()
# plt.pcolormesh(lons,lats,elevs,cmap='terrain')
# plt.colorbar()
# plt.show()