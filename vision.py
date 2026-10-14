"""2026 portability repairs; architectures retain the submitted coursework."""
from pathlib import Path
import numpy as np
from PIL import Image

def load_images(directory,classes=None,size=(71,71)):
    """2026 manifest-based loader; explicit labels survive missing test classes."""
    root=Path(directory)
    discovered={entry.name for entry in root.iterdir() if entry.is_dir()}
    labels=sorted(discovered) if classes is None else list(classes)
    if not labels or len(set(labels))!=len(labels): raise ValueError('nonempty unique classes required')
    if any(not isinstance(label,str) or not label or label in ('.','..') or '/' in label or '\\' in label for label in labels):
        raise ValueError('classes must be directory names')
    if discovered-set(labels): raise ValueError('unknown class in evaluation data')
    if len(size)!=2 or any(not isinstance(value,int) or value<=0 for value in size):
        raise ValueError('positive integer image dimensions required')
    ids={label:index for index,label in enumerate(labels)}
    records=sorted((path for path in root.glob('*/*') if path.is_file()
                    and path.parent.name in ids and path.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')),
                   key=lambda path:(ids[path.parent.name],path.name))
    if not records: raise ValueError('no images found')
    features=np.empty((len(records),size[0]*size[1]),dtype=np.float32)
    targets=np.fromiter((ids[path.parent.name] for path in records),dtype=np.int64)
    for index,path in enumerate(records):
        with Image.open(path) as image:
            features[index]=np.asarray(image.convert('L').resize(size),dtype=np.float32).reshape(-1)/255
    return features,targets,labels

def fit_svm(X,y,*,kernel='rbf'):
    from sklearn.svm import SVC
    # Original RBF SVC, no test-derived scaling or cross-split fitting.
    if len(X)!=len(y) or len(set(y))<2: raise ValueError('at least two labeled classes required')
    if kernel not in ('linear','rbf'): raise ValueError('expected linear or rbf kernel')
    return SVC(kernel=kernel).fit(X,y)

def preprocess_rgb(images,transfer=False):
    arr=np.asarray(images)
    if arr.ndim!=4 or arr.shape[-1]!=3 or not np.all(np.isfinite(arr)):
        raise ValueError('finite batch of RGB images required')
    if np.min(arr)<0 or np.max(arr)>255: raise ValueError('expected pixel range [0,255]')
    arr=arr.astype(np.float32)
    if transfer:
        from tensorflow.keras.applications.xception import preprocess_input
        return preprocess_input(arr)
    return arr/255

def generators(train_dir,test_dir,*,transfer=False,seed=0):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.xception import preprocess_input
    classes=sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())
    if len(classes)<2: raise ValueError('at least two training classes required')
    test_names={p.name for p in Path(test_dir).iterdir() if p.is_dir()}
    if test_names-set(classes): raise ValueError('unknown test class')
    options={'preprocessing_function':preprocess_input} if transfer else {'rescale':1/255}
    # Validation comes only from training directory; test is final evaluation only.
    training=ImageDataGenerator(validation_split=.2,**options)
    kw=dict(target_size=(71,71),color_mode='rgb',batch_size=10,class_mode='sparse',classes=classes,seed=seed)
    train=training.flow_from_directory(train_dir,subset='training',shuffle=True,**kw)
    val=training.flow_from_directory(train_dir,subset='validation',shuffle=False,**kw)
    test=ImageDataGenerator(**options).flow_from_directory(test_dir,shuffle=False,**kw)
    if not train.samples or not val.samples or not test.samples: raise ValueError('nonempty train/validation/test required')
    return train,val,test

def build_model(kind,classes,*,weights=None):
    import tensorflow as tf
    L=tf.keras.layers
    if not isinstance(classes,int) or classes<2: raise ValueError('at least two classes required')
    if kind=='mlp': layers=[L.Input((71,71,3)),L.Flatten(),L.Dense(16,activation='relu'),L.Dense(32,activation='relu')]
    elif kind=='cnn':
        layers=[L.Input((71,71,3))]
        for filters in (16,32,64): layers.extend([L.Conv2D(filters,3,activation='relu',padding='valid'),L.AveragePooling2D(2)])
        layers.extend([L.Flatten(),L.Dense(128,activation='relu'),L.Dense(64,activation='relu')])
    elif kind=='custom':
        layers=[L.Input((71,71,3)),L.Conv2D(32,3,activation='relu'),L.Conv2D(64,3,activation='relu'),L.MaxPooling2D(2),L.Dropout(.25),L.Flatten(),L.Dense(128,activation='relu'),L.Dropout(.5)]
    elif kind=='xception':
        base=tf.keras.applications.Xception(include_top=False,weights=weights,input_shape=(71,71,3));base.trainable=False
        layers=[base,L.AveragePooling2D(2),L.Flatten(),L.Dense(256,activation='relu'),L.Dropout(.4)]
    else: raise ValueError('unknown architecture')
    model=tf.keras.Sequential([*layers,L.Dense(classes,activation='softmax')])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    return model
