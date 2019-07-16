import tensorflow as tf

def preprocess_image(image):
    image = tf.image.decode_image(image, channels=3)
    image_size = tf.shape(image)
    #image = tf.image.resize_images(image, [331, 331])
    #image = tf.image.central_crop(image,0.9)
    
    def f1(): return tf.image.crop_to_bounding_box(
                        image,
                        0,
                        tf.cast((image_size[1]-image_size[0])/3,tf.int32),
                        image_size[0],
                        image_size[0]+tf.cast((image_size[1]-image_size[0])/4,tf.int32))
    def f2(): return tf.image.crop_to_bounding_box(
                        image,
                         tf.cast((image_size[0]-image_size[1])/3,tf.int32),
                        0,
                        image_size[1]+tf.cast((image_size[0]-image_size[1])/4,tf.int32),
                        image_size[1])
    
    image = tf.cond(
        image_size[0] > image_size[1],
        true_fn=f2,
        false_fn=f1)

    image = tf.image.resize_images(image, [299, 299],align_corners=True,method=1)

    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(
                image,0.125) #0.5247078
    image = tf.image.random_saturation(image,0.5,1.5) #0.3824261 and 1.4029386
    image = tf.image.random_hue(image,0.2) #-0.1267652 and 0.1267652
    image = tf.image.random_contrast(image,0.5,1.5) #0.3493415 and 1.3461331
    return image

def load_and_preprocess_image(path,is_training):
    image = tf.read_file(path)
    image = preprocess_image(image)
    if is_training:
        image = augment_image(image)
    #image /= 255.0  # normalize to [0,1] range
    image = 2*tf.cast(image,tf.float32)/255.0 -1

    return image

def tfdata_generator(all_image_path, labels=None, is_training=False,buffer_size=300,batch_size=32):
    '''Construct a data generator using `tf.Dataset`. '''

    dataset = tf.data.Dataset.from_tensor_slices(all_image_path)
    dataset = dataset.map(lambda x: load_and_preprocess_image(x,is_training))

    if labels is not None:
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
        dataset = tf.data.Dataset.zip((dataset, label_ds))
    if is_training:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset
