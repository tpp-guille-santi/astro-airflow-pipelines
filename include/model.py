import pathlib

import tensorflow
from tensorflow import keras

from include.repositories import MinioRepository
from settings import settings


def prepare_dataset(data_dir, subset):
    return tensorflow.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        validation_split=settings.VALIDATION_SPLIT,
        subset=subset,
        seed=settings.SEED,
        batch_size=settings.BATCH_SIZE,
        image_size=(settings.IMG_HEIGHT, settings.IMG_WIDTH),
    )


def train_and_evaluate_model(minio_repository: MinioRepository) -> tuple[keras.Model, float]:
    train_dataset = minio_repository.prepare_minio_dataset('training')
    validation_dataset = minio_repository.prepare_minio_dataset('validation')

    num_classes = len(train_dataset.class_names)

    val_batches = tensorflow.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    train_dataset = train_dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    # Create the base model from the pre-trained model MobileNet V2
    img_shape = (settings.IMG_HEIGHT, settings.IMG_WIDTH) + (3,)
    base_model = tensorflow.keras.applications.MobileNetV2(
        input_shape=img_shape, include_top=False, weights='imagenet'
    )

    base_model.trainable = False

    data_augmentation = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.RandomFlip(
                'horizontal', input_shape=(settings.IMG_HEIGHT, settings.IMG_WIDTH, 3)
            ),
            tensorflow.keras.layers.RandomRotation(0.1),
            tensorflow.keras.layers.RandomZoom(0.1),
        ]
    )
    preprocess_input = tensorflow.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tensorflow.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tensorflow.keras.layers.Dense(num_classes, activation='softmax')

    inputs = tensorflow.keras.Input(shape=(settings.IMG_HEIGHT, settings.IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tensorflow.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tensorflow.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE),
        loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_dataset, epochs=settings.INITIAL_EPOCHS, validation_data=validation_dataset
    )

    base_model.trainable = True

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[: settings.FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=settings.LEARNING_RATE / 10),
        metrics=['accuracy'],
    )

    total_epochs = settings.INITIAL_EPOCHS + settings.FINE_TUNE_EPOCHS

    model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset)

    return model, test_accuracy


# TODO:
#  Remove this function in the future.
#  It's only used so that we don't have to train the model every time we want to run the pipeline.
def load_and_evaluate_model():
    data_dir = pathlib.Path(settings.IMAGES_PATH)
    validation_dataset = prepare_dataset(data_dir, 'validation')
    val_batches = tensorflow.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    model = tensorflow.keras.models.load_model(settings.TRAINED_MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    return model, test_accuracy
