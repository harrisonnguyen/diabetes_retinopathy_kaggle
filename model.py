from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.xception import Xception
# create the base pre-trained model
import preprocess
def main():
    df = pd.read_csv("dataset/training-labels.csv")
    df['Filename'] = 'dataset/output_combined2/' + df['Filename'].astype(str)

    generator = preprocess.tfdata_generator(sub_df['Filename'].values,
                                        sub_df['Drscore'].values,
                                        is_training=True)

    base_model = Xception(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(
        generator,
        steps_per_epoch=len(sub_df['Filename'].values)//32,
        epochs=5,
        verbose = 1)
