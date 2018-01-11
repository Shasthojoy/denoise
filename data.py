import tensorflow as tf
import random

IMG_WIDTH = 800
IMG_HEIGHT = 600
CROP_SIZE = 128
NUM_CROPS_PER_IMAGE = 100


def input_fn(records, shuffle=False, repeat_count=1, take_one=False):
    def _parse_function(example_proto):
        features = {
            "color": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "color4096": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "normal": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "depth": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "gloss": tf.FixedLenFeature(
                (), tf.string, default_value="")}

        parsed_features = tf.parse_single_example(example_proto, features)

        tmp = tf.decode_raw(
            parsed_features["color4096"], tf.float16, little_endian=True)
        color_final = tf.image.convert_image_dtype(
            tf.reshape(tmp, (IMG_HEIGHT, IMG_WIDTH, 3)), dtype=tf.float32)

        tmp = tf.decode_raw(
            parsed_features["color"], tf.float16, little_endian=True)
        color = tf.image.convert_image_dtype(
            tf.reshape(tmp, (IMG_HEIGHT, IMG_WIDTH, 3)), dtype=tf.float32)

        tmp = tf.decode_raw(
            parsed_features["depth"], tf.float16, little_endian=True)
        depth = tf.image.convert_image_dtype(
            tf.reshape(tmp, (IMG_HEIGHT, IMG_WIDTH, 1)), dtype=tf.float32)

        tmp = tf.image.decode_jpeg(parsed_features["normal"], channels=3)
        normal = tf.image.convert_image_dtype(
            tf.reshape(tmp, (IMG_HEIGHT, IMG_WIDTH, 3)), dtype=tf.float32)

        tmp = tf.image.decode_jpeg(parsed_features["gloss"], channels=1)
        gloss = tf.image.convert_image_dtype(
            tf.reshape(tmp, (IMG_HEIGHT, IMG_WIDTH, 1)), dtype=tf.float32)

        return ((tf.concat([
            color,
            depth,
            normal,
            gloss], axis=2)),
            (color_final))

    def _crop_function(example, target):
        cropped_examples = []
        cropped_targets = []
        for i in range(NUM_CROPS_PER_IMAGE):
            x = random.randint(0, IMG_WIDTH - CROP_SIZE)
            y = random.randint(0, IMG_HEIGHT - CROP_SIZE)
            cropped_examples.append(
                tf.image.crop_to_bounding_box(
                    example, y, x, CROP_SIZE, CROP_SIZE))
            cropped_targets.append(
                tf.image.crop_to_bounding_box(
                    target, y, x, CROP_SIZE, CROP_SIZE))
        return tf.data.Dataset.from_tensor_slices(
            (tf.stack(cropped_examples), tf.stack(cropped_targets)))

    dataset = tf.data.TFRecordDataset(records).map(_parse_function)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=512)

    dataset = dataset.flat_map(_crop_function).repeat(repeat_count).batch(128)

    if take_one:
        dataset = dataset.take(1)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
