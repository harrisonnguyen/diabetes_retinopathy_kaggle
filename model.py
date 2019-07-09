from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.xception import Xception
# create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False)
