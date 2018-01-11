import sys
import os
import io
import OpenEXR
import Imath
import numpy as np
import tensorflow as tf
from scipy import ndimage
from PIL import Image


def print_usage():
    print('''USAGE:
        CONVERT AOV DATA INTO TFRECORD
        aovtool.py convert aov_folder cam_min_idx cam_max_idx sample_count
        out_tfrecord_file
        WHERE
        aov_folder - folder containing AOV data
        cam_min_idx - minimum camera index
        cam_max_idx - maximum camera index
        sample_count - sample count to use for an example image
        out_tfrecord_file - output TFRecordFile
        OR
        SHOW PATICULAR PIECE OF DATA IN TFRECORD FILE
        aovtool.py test in_tfrecord_file record_idx image type w h c
        in_tfrecord_file - input file
        record_idx - index of the record to show
        image - image data to show (color, depth, normal, gloss, etc)
        type - type of the data (raw or jpeg)
        w - image width
        h - image height
        c - number of channels''')


def load_exr(filename, mode="RGB"):
    # Load OpenEXR image and return it as a numpy array
    exr_file = OpenEXR.InputFile(filename)
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)
    dw = exr_file.header()['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    data = np.ndarray(shape=(h, w, len(mode)), dtype=np.float16)
    for i, c in enumerate(mode):
        data[:, :, i] = np.reshape(
            np.fromstring(
                exr_file.channel(c, pixel_type),
                dtype=np.float16), newshape=(h, w))
    if np.any(np.isnan(data)):
        print("NaNs detected: ", filename)
        data[np.isnan(data)] = 0.0
    if np.any(np.isinf(data)):
        print("Inf detected: ", filename)
        data[np.isinf(data)] = 0.0
    return data


def load_jpg(filename):
    img = ndimage.imread(filename, mode="RGB").astype(np.float16)
    return img / 255.0


def load_binary(filename):
    with open(filename, 'rb') as f:
        return f.read()


def divide_non_zero(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
    return c


def demodulate_albedo(color, albedo):
    return divide_non_zero(color, albedo)


def normlaize_depth(depth):
    min = np.amin(depth)
    max = np.amax(depth)
    return depth / (max - min)


def show_hdr_image(image):
    h, w, c = image.shape
    rgb = np.zeros((h, w, 3), 'uint8')
    for i in range(c):
        rgb[:, :, i] = np.clip(image[:, :, i] * 255, 0, 255)
    im = Image.fromarray(rgb, "RGB")
    im.show()


def aovs2tfrecord(aov_folder, tfr_name, samplecnt, cam_min=1, cam_max=500):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfr_name)

    record_count = 0
    for i in range(cam_min, cam_max):
        aov_name = os.path.join(aov_folder, "cam_" + str(i))
        cnt_str = str(samplecnt)

        if not os.path.exists(aov_name + "_aov_color_f" + cnt_str + ".exr"):
            continue

        try:
            feature = dict()

            albedo = load_jpg(aov_name +
                              "_aov_albedo_f" + cnt_str + ".jpg")
            color = load_exr(aov_name +
                             "_aov_color_f" + cnt_str + ".exr")
            color4096 = load_exr(aov_name + "_aov_color_f4096.exr")

            color = demodulate_albedo(color, albedo)
            color4096 = demodulate_albedo(color4096, albedo)

            # Color and final color
            feature["color"] = _bytes_feature(
                tf.compat.as_bytes(color.tostring()))
            feature["color4096"] = _bytes_feature(
                tf.compat.as_bytes(color4096.tostring()))

            # Depth
            depth = load_exr(aov_name +
                             "_aov_view_shading_depth_f" +
                             cnt_str + ".exr",
                             mode="R")
            depth = normlaize_depth(depth)
            feature["depth"] = _bytes_feature(
                tf.compat.as_bytes(depth.tostring()))

            # Normal
            img = load_binary(aov_name +
                              "_aov_view_shading_normal_f" +
                              cnt_str + ".jpg")
            feature["normal"] = _bytes_feature(tf.compat.as_bytes(img))
            img = load_binary(aov_name + "_aov_gloss_f" +
                              cnt_str + ".jpg")

            # Gloss
            feature["gloss"] = _bytes_feature(tf.compat.as_bytes(img))

            example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            record_count += 1

        except Exception as err:
            print("Error:", err)
            continue

    print("Total records in", tfr_name, ":", record_count)
    return


def enumerate_records(name):
    record = dict()
    iterator = tf.python_io.tf_record_iterator(path=name)
    for record in iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        yield example


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: wrong number of arguments")
        print_usage()
        sys.exit(-1)

    mode = sys.argv[1]

    if mode != "convert" and mode != "test":
        print("Wrong mode, only convert or test modes supported")

    if mode == "convert":
        if len(sys.argv) < 6:
            print("ERROR: wrong number of parameters")
            print_usage()
            sys.exit(-1)

        folder = sys.argv[2]
        cam_min = int(sys.argv[3])
        cam_max = int(sys.argv[4])
        sample_count = int(sys.argv[5])
        tfrec = sys.argv[6]
        print("Running conversion...")
        aovs2tfrecord(folder, tfrec, sample_count, cam_min, cam_max)
    else:
        if len(sys.argv) < 9:
            print("ERROR: wrong number of parameters")
            print_usage()
            sys.exit(-1)

        tfrec = sys.argv[2]
        idx = int(sys.argv[3])
        data = sys.argv[4]
        dtype = sys.argv[5]
        w = int(sys.argv[6])
        h = int(sys.argv[7])
        c = int(sys.argv[8])

        print("Running test...")
        cnt = 0
        for record in enumerate_records(tfrec):
            if cnt == idx:
                bytes = record.features.feature[data].bytes_list.value[0]
                if dtype == "raw":
                    data = np.reshape(
                        np.fromstring(bytes, dtype=np.float16),
                        newshape=(h, w, c))
                    show_hdr_image(data)
                if dtype == "jpeg":
                    stream = io.BytesIO(bytes)
                    data = load_jpg(stream)
                    picture = Image.open(stream)
                    picture.show()
            cnt += 1
