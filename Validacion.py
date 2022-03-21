import numpy as np
import gdal
import ogr
from sklearn import metrics

naip_fn = 'E:/temp/naip/clipped.tif'

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)

test_fn = 'E:/temp/naip/test.shp'
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()

# Crear nuevo raster en memoria
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())

#Rasterizar los puntos de entrenamiento
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open('E:/temp/naip/clasificacion.tif')
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)
cm = metrics.confusion_matrix(truth[idx], pred[idx])

# Exactitud por pixeles
print (cm)

print(cm.diagonal())
print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)
