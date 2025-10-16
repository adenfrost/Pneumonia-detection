# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import DenseNet121
# from src.utils import DicomSequence
# import glob
# from sklearn.model_selection import train_test_split

# pneum_files = glob.glob('data/train/PNEUMONIA/*.jpeg')
# norm_files = glob.glob('data/train/NORMAL/*.jpeg')
# filepaths = pneum_files + norm_files
# labels = [1]*len(pneum_files) + [0]*len(norm_files)



# train_fp, val_fp, train_y, val_y = train_test_split(filepaths, labels, test_size=0.15, stratify=labels, random_state=42)

# train_gen = DicomSequence(train_fp, train_y)
# val_gen = DicomSequence(val_fp, val_y, shuffle=False)

# base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')
# base.trainable = False

# inp = layers.Input(shape=(224,224,3))
# x = base(inp, training=False)
# x = layers.Dropout(0.3)(x)
# out = layers.Dense(1, activation='sigmoid')(x)
# model = models.Model(inputs=inp, outputs=out)

# model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
#               loss='binary_crossentropy',
#               metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint('saved_models/densenet_pneumonia.h5', save_best_only=True, monitor='val_auc', mode='max'),
#     tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')
# ]

# model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from src.utils import DicomSequence
import glob
from sklearn.model_selection import train_test_split

pneum_files = glob.glob('data/train/PNEUMONIA/*.jpeg')
norm_files = glob.glob('data/train/NORMAL/*.jpeg')
filepaths = pneum_files + norm_files
labels = [1]*len(pneum_files) + [0]*len(norm_files)

train_fp, val_fp, train_y, val_y = train_test_split(
    filepaths, labels, test_size=0.15, stratify=labels, random_state=42
)

train_gen = DicomSequence(train_fp, train_y)
val_gen = DicomSequence(val_fp, val_y, shuffle=False)

# KEY FIX: Don't use pooling='avg' - keep spatial dimensions
base = DenseNet121(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3),
    pooling=None  # CHANGED: Don't apply pooling yet
)
base.trainable = False

# Build model with manual pooling after the base
inp = layers.Input(shape=(224, 224, 3))
x = base(inp, training=False)  # Output shape: (batch, 7, 7, 1024)

# Now apply global average pooling explicitly
x = layers.GlobalAveragePooling2D()(x)  # (batch, 1024)

# Then apply dropout and dense layer
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inp, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'saved_models/densenet_pneumonia.h5',
        save_best_only=True,
        monitor='val_auc',
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max'
    )
]

model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)