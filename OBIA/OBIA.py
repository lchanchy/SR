import numpy as np
import gdal
import scipy
from skimage import exposure
from skimage.segmentation import slic
import ogr
import time
from sklearn.ensemble import RandomForestClassifier


naip_fn = 'E:/temp/naip/clipped.tif'

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount
band_data = []
print('bands', naip_ds.RasterCount, 'rows', naip_ds.RasterYSize, 'columns', naip_ds.RasterXSize)
for i in range(1, nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
print(band_data.shape)

#Escalar los valores de la imagen de 0.0 a 1
img = exposure.rescale_intensity(band_data)

#Ejemplos de segmentacion quickshift y slic
seg_start = time.time()
#segments = quickshift(img, convert2lab=False) metodo quickshift
segments = slic(img, n_segments=50000, compactness=0.1, start_label=1) # metodo slic
print('segmentos completados', time.time() - seg_start)

#Caracteristicas espectrales
def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # en este caso la varianza = nan, cambiela 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features

obj_start = time.time()
segment_ids = np.unique(segments)
objects = []
object_ids = []
for id in segment_ids:
    segment_pixels = img[segments == id]
    print('pixeles por id', id, segment_pixels.shape)
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print('creado', len(objects), 'objetos con', len(objects[0]), 'variables en',
      time.time() - obj_start, 'segundos')

#Guardar informacion
segments_fn = 'E:/temp/prueba/segmentos_slic1.tif'
segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
segments_ds.SetProjection(naip_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None

#Leer de datos de entrenamiento
train_fn = 'E:/temp/naip/train.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

# Crear nuevo raster en memoria
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())

#Rasterizar los puntos de entrenamiento
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

# Recuperar datos rasterizados e imprimir estadísticas básicas
data = target_ds.GetRasterBand(1).ReadAsArray()
print('min', data.min(), 'max', data.max(), 'mean', data.mean())

# Obtener los segmentos para cada clase de cobertura
#Ninguna clase de segmento puede presentar mas de una clase

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]
print('class values', classes)

segments_per_class = {}

for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print("Segmentos de Entrenamiento por clase", klass, ":", len(segments_of_class))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"

#Clasificacion con Random Forest ML

train_img = np.copy(segments)
threshold = train_img.max() + 1

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label

# Clasificar la imagen

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print('Objetos de entrenamiento para clase', klass, ':', len(class_train_object))

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
print('Ajuste del clasificador de bosque aleatorio')
predicted = classifier.predict(objects)
print('Clasificaciones de predicción')

clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass

print('Predicción aplicada a una matriz numpy')

mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0

print('Guardar clasificación a raster con gdal')

clfds = driverTiff.Create('E:/temp/prueba/clasificacion.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print('HECHO!')




