import tensorflow as tf
import numpy as np
# using tensorflow dataset utils to make things work with TPUs

def build_decoder(with_labels=True, target_size=(480,270),ext='png'):
    def decode(path):
        file_bytes = tf.io.read_file(path)

        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0 # Normalization
        img = tf.image.resize(img, target_size)


        return img

    def decode_with_labels(path, label):
        return decode(path), label
    
    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=128, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    print(np.array(labels.shape))
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)
    
    return dset

def make_train_test_split(df, test_size=0.22):
    '''
    Create train, test
    split for dataframes..
    '''
    splits = []
    train_num = len(df) - math.ceil(len(df)*test_size)
    for i in range(len(df)):
        if i < train_num:
            splits.append('train')
        else:
            splits.append('val')

    df['split'] = splits
    # ---------------
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    train_df = df.loc[df['split'] == 'train']
    valid_df = df.loc[df['split'] == 'val']

    del df
    
    return train_df, valid_df

