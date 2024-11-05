import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Flatten, MaxPooling2D, Input, GlobalAveragePooling1D
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
import tensorflow_addons as tfa
import pickle as pkl
from tensorflow.keras.preprocessing import image
from tensorflow_addons.optimizers import AdamW

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units,
        })
        return config

def create_vit_model(input_shape, patch_size, num_patches, projection_dim, transformer_units, transformer_layers, model_name, num_heads, mlp_head_units, num_classes, include_localization=False):
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    class_logits = layers.Dense(num_classes, activation='softmax', name='classification')(features)

    # Localization head (optional)
    if include_localization:
        localization_features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        localization_logits = layers.Dense(1, activation='sigmoid', name='localization')(localization_features)
        model = keras.Model(inputs=inputs, outputs=[class_logits, localization_logits], name=model_name)
    else:
        model = keras.Model(inputs=inputs, outputs=class_logits, name=model_name)

    return model

def model_maker(target_size, model_id, num_classes=3, include_localization=False):
    image_size = target_size[0]
    patch_size = 10
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ]
    transformer_layers = 2
    mlp_head_units = [50, 50]

    model = create_vit_model((*target_size, 3),
                             patch_size,
                             num_patches,
                             projection_dim,
                             transformer_units,
                             transformer_layers,
                             'PlantDiseaseClassifier_VIT_' + str(model_id),
                             num_heads, mlp_head_units, num_classes,
                             include_localization=include_localization)
    model.summary()
    return model

def gen_maker(train_path, val_path, target_size=(100, 100), batch_size=16, mode='categorical'):
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=mode
    )

    validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode=mode
    )

    return train_generator, validation_generator

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, model_path, model_id, include_localization=False):
        super(CustomCallback, self).__init__()
        self.val_gen = val_gen
        self.model_path = model_path
        self.model_id = model_id
        self.include_localization = include_localization

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path + 'epoch{}-id{}'.format(epoch, self.model_id))
        y_pred, y_pred_loc = self.model.predict(self.val_gen)
        y_true = self.val_gen.classes

        # Classification metrics
        y_pred_class = np.argmax(y_pred, axis=1)
        cls_report = classification_report(y_true, y_pred_class)
        cnf = confusion_matrix(y_true, y_pred_class)
        print('\nClassification report:\n', cls_report)
        print('\nConfusion matrix:\n', cnf)

        # Localization metrics (if applicable)
        if self.include_localization:
            y_true_loc = self.val_gen.labels
            loc_mse = mean_squared_error(y_true_loc, y_pred_loc)
            print('\nLocalization MSE:', loc_mse)

def main(include_localization=False):
    epochs = 1
    init_epoch = 0
    train_dir = '/content/drive/MyDrive/plant/train'
    val_dir = '/content/drive/MyDrive/plant/val'
    model_id = 1
    load_model = 0
    load_path = None
    backup_path = '/content/'
    batch_size = 16
    mode = 'categorical'
    target_size = 100

    # Generate training and validation datasets
    train_gen, val_gen = gen_maker(
        train_dir,
        val_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        mode=mode
    )

    # Custom callback
    clbk = CustomCallback(val_gen, backup_path, model_id, include_localization=include_localization)

    # Set learning rate and weight decay
    learning_rate = 0.001
    weight_decay = 0.0001

    # Create the model
    model = model_maker((target_size, target_size), model_id, include_localization=include_localization)

    # Set up optimizer
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Compile model
    if include_localization:
        model.compile(
            optimizer=optimizer,
            loss={'classification': categorical_crossentropy, 'localization': binary_crossentropy},
            metrics={'classification': 'acc', 'localization': 'mse'}
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy,
            metrics=['acc']
        )

    # Load pre-trained model if specified
    if load_model:
        model = tf.keras.models.load_model(load_path)

    # Train the model
    results = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[clbk],
        initial_epoch=init_epoch
    )

    # Predict validation data
    if include_localization:
        y_pred_valid, y_pred_loc_valid = model.predict(val_gen)
    else:
        y_pred_valid = model.predict(val_gen)

    # Save history and model
    # history = {
    #     'train loss': results.history['loss'],
    #     'val loss': results.history['val_loss'],
    #     'train acc': results.history['acc'],
    #     'val acc': results.history['val_acc'],
    #     'y_true_valid': val_gen.classes,
    #     'y_pred_valid': y_pred_valid,
    #     'id': model_id
    # }

    # if include_localization:
    #     history['y_pred_loc_valid'] = y_pred_loc_valid

    model.save(f"{backup_path}model.h5")
    print('Saved')
    # with open(f"{backup_path}history-id-{model_id}.pkl", 'wb') as f:
    #     pkl.dump(history, f)

    # Plot training and validation metrics
    # plt.subplots(figsize=(15, 15))
    # plt.subplot(2, 1, 1)
    # plt.plot(results.history['loss'], '-', color=[0, 0, 1, 1])
    # plt.plot(results.history['val_loss'], '-', color=[1, 0, 0, 1])
    # plt.legend(['train loss', 'val loss'])

    # plt.subplot(2, 1, 2)
    # plt.plot([0, *results.history['acc']], '-', color=[0, 0, 1, 1])
    # plt.plot([0, *results.history['val_acc']], '-', color=[1, 0, 0, 1])
    # plt.legend(['train acc', 'val acc'])

    # plt.savefig(f"{backup_path}charts.png")

# Train the model with localization
main(include_localization=True)

# Evaluate the model with localization
model = tf.keras.models.load_model(
    '/content/model.h5',
    custom_objects={
        'AdamW': AdamW,
        'Patches': Patches,
        'PatchEncoder': PatchEncoder  # Add PatchEncoder here
    }
)

img_path = '/content/drive/MyDrive/plant/val/Healthy/Healthy1030.jpg'
img_path = '/content/drive/MyDrive/plant/val/Yellow_rust/Yellow_rust486.jpg'
img_path = '/content/drive/MyDrive/plant/val/Brown_rust/Brown_rust036.jpg'
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

class_preds, loc_preds = model.predict(img_array)
print("Predicted class:", np.argmax(class_preds))
print("Localization prediction:", loc_preds[0][0])

import cv2
def draw_bounding_boxes(image, boxes, scores, labels):
    """
    Draw bounding boxes on the given image.
    """
    img = image.copy()
    # for box, score, label in zip(boxes, scores, labels):
    #     print(box)
    #     x1, y1, x2, y2 = [int(x) for x in box]
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
    #     cv2.putText(img, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    # make boxes as int array
    boxes = np.array(boxes).astype(int)
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    score = scores[0]
    label = labels[0]
    print(x1, y1, x2, y2, score, label)
    cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
    # cv2.putText(img, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    return img
boxes = loc_preds[0] * [100, 100, 100, 100]  # Rescale the bounding box coordinates
boxes = [b for b in boxes]
print(boxes)
labels = ['Disease']
scores = [class_preds[0, np.argmax(class_preds)]]
# convert img to numpy array
img = np.array(img)
img_with_boxes = draw_bounding_boxes(img, boxes, scores, labels)
plt.imshow(img_with_boxes)
plt.show()