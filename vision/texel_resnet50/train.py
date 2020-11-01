import tensorflow as tf

path = "/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/texel_data"

datagen = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    labels='inferred',
    label_mode='binary',
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)

