
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Flatten, MaxPooling2D, Input
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow_addons as tfa
import pickle as pkl
from tensorflow.keras.preprocessing import image
from tensorflow_addons.optimizers import AdamW


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
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

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, transformer_units, transformer_layers, model_name, num_heads, mlp_head_units, num_classes):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    #representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation = 'softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits, name = model_name)
    return model

def model_maker(target_size, model_id, num_classes = 3):
    """ This function creates a trainable model.
        params:
            target_size: tuple, size of the input image to the network
            model_id: integer, it can be 1 to 4
        returns:
            tensorflow trainable model.
    """


    image_size = target_size[0]  # We'll resize input images to this size
    patch_size = 10  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim]  # Size of the transformer layers
    transformer_layers = 2
    mlp_head_units = [50, 50]
    model = create_vit_classifier((*target_size, 3),
                                  patch_size,
                                  num_patches,
                                  projection_dim,
                                  transformer_units,
                                  transformer_layers,
                                  'WheatClassifier_VIT_'+str(model_id),
                                  num_heads, mlp_head_units, num_classes)
    model.summary()
    return model

def gen_maker(train_path, val_path, target_size=(100, 100), batch_size=16, mode='categorical'):
    """
    This function creates data generators for train and validation data.
    params:
        train_path: path to the training data folder, string.
        val_path: path to the validation data folder, string.
        target_size: size of the inputs to the network, tuple.
        batch_size: the batch size for training and validation, integer.
        mode: classification mode, it can be either "binary" or "categorical"
    returns:
        train_generator: data generator for training data.
        validation_generator: data generator for validation data.
    """

    train_datagen = ImageDataGenerator( rotation_range=10,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    channel_shift_range=0.0,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=mode)

    validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode=mode)
    return train_generator, validation_generator

class CustomCallback(tf.keras.callbacks.Callback):
    """
    This callback saves the model at the end of each epoch and calculates
    the confusion matrix and classification report on the validation data.
    """

    def __init__(self, val_gen, model_path, model_id):

        super(CustomCallback, self).__init__()
        self.val_gen = val_gen
        self.model_path = model_path
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path + 'epoch{}-id{}'.format(epoch,self.model_id ))
        y_pred = self.model.predict(self.val_gen)
        y_pred = np.squeeze(np.argmax(y_pred, axis = 1))
        y_true = self.val_gen.classes
        cnf = confusion_matrix(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred)
        print('\nclassification report:\n', cls_report)
        print('\nconfusion matrix:\n', cnf)


def gen_maker(train_path, val_path, target_size=(100, 100), batch_size=16, mode='categorical'):
    """
    This function creates data generators for train and validation data.
    params:
        train_path: path to the training data folder, string.
        val_path: path to the validation data folder, string.
        target_size: size of the inputs to the network, tuple.
        batch_size: the batch size for training and validation, integer.
        mode: classification mode, it can be either "binary" or "categorical"
    returns:
        train_generator: data generator for training data.
        validation_generator: data generator for validation data.
    """

    train_datagen = ImageDataGenerator( rotation_range=10,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    channel_shift_range=0.0,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=mode)

    validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode=mode)
    return train_generator, validation_generator

class CustomCallback(tf.keras.callbacks.Callback):
    """
    This callback saves the model at the end of each epoch and calculates
    the confusion matrix and classification report on the validation data.
    """

    def __init__(self, val_gen, model_path, model_id):

        super(CustomCallback, self).__init__()
        self.val_gen = val_gen
        self.model_path = model_path
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path + 'epoch{}-id{}'.format(epoch,self.model_id ))
        y_pred = self.model.predict(self.val_gen)
        y_pred = np.squeeze(np.argmax(y_pred, axis = 1))
        y_true = self.val_gen.classes
        cnf = confusion_matrix(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred)
        print('\nclassification report:\n', cls_report)
        print('\nconfusion matrix:\n', cnf)

epochs = 1                # Number of total epochs
init_epoch = 0              # Initial epoch
train_dir = '/content/train/'  # Training data folder path
val_dir = '/content/val/'      # Validation data folder path
model_id = 1                # Model ID: it can be 1, 2, 3, or 4
load_model = 0              # If 1, load a previously trained model
load_path = None            # Path to the pre-trained model
backup_path = '/content/'   # Path to store the model
batch_size = 16             # Batch size for training
mode = 'categorical'        # Classification mode
target_size = 100           # Size of the input images

train_dir = '/content/drive/MyDrive/plant/train'
val_dir = '/content/drive/MyDrive/plant/val'

# Define the main training function
def main():

    # Generate training and validation datasets
    train_gen, val_gen = gen_maker(
        train_dir,
        val_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        mode=mode
    )

    # Custom callback
    clbk = CustomCallback(val_gen, backup_path, model_id)

    # Set learning rate and weight decay
    learning_rate = 0.001
    weight_decay = 0.0001

    # Create the model
    model = model_maker((target_size, target_size), model_id)

    # Set up optimizer
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Compile model
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
    y_pred_valid = model.predict(val_gen)

    # Save history and model
    history = {
        'train loss': results.history['loss'],
        'val loss': results.history['val_loss'],
        'train acc': results.history['acc'],
        'val acc': results.history['val_acc'],
        'y_true_valid': val_gen.classes,
        'y_pred_valid': y_pred_valid,
        'id': model_id
    }
    model.save(f"{backup_path}model.h5")

    with open(f"{backup_path}history-id-{model_id}.pkl", 'wb') as f:
        pkl.dump(history, f)

    # Plot training and validation metrics
    plt.subplots(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(results.history['loss'], '-', color=[0, 0, 1, 1])
    plt.plot(results.history['val_loss'], '-', color=[1, 0, 0, 1])
    plt.legend(['train loss', 'val loss'])

    plt.subplot(2, 1, 2)
    plt.plot([0, *results.history['acc']], '-', color=[0, 0, 1, 1])
    plt.plot([0, *results.history['val_acc']], '-', color=[1, 0, 0, 1])
    plt.legend(['train acc', 'val acc'])

    plt.savefig(f"{backup_path}charts.png")

# Run main function
main()

# EVALUATE THE MODEL

# Load the model with the custom optimizer specified
model = tf.keras.models.load_model('/content/model.h5', custom_objects={'AdamW': AdamW})

# Load and preprocess a single image
img_path = '/content/drive/MyDrive/plant/val/Brown_rust/Brown_rust036.jpg'
img = image.load_img(img_path, target_size=(100, 100))  # Resize as per model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize if necessary

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)



