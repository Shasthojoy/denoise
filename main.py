import data
import model
# import numpy as np
import tensorflow as tf
# from PIL import Image

NUM_EPOCHS = 30

tf.logging.set_verbosity(tf.logging.INFO)

estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    model_dir="./model",
    params={"learning_rate": 1e-4})

datasets = [
    "dataset/SponzaDay8.tfrecord"
]

train = True
if train:
    for i in range(NUM_EPOCHS):
        print("Starting epoch", i)
        estimator.train(input_fn=lambda: data.input_fn(datasets, True, 1))

# obj = estimator.predict(
#     input_fn=lambda: data.input_fn1("KitchenEveningEval.tfrecord", 0))

# for p in obj:
#     rgb = np.zeros((1024, 1024, 3), 'uint8')
#     rgb[..., 0] = np.clip(
#         np.power(1.5 * p["orig"][:, :, 0], 1.0 / 2.2) * 255, 0, 255)
#     rgb[..., 1] = np.clip(
#         np.power(1.5 * p["orig"][:, :, 1], 1.0 / 2.2) * 255, 0, 255)
#     rgb[..., 2] = np.clip(
#         np.power(1.5 * p["orig"][:, :, 2], 1.0 / 2.2) * 255, 0, 255)

#     im = Image.fromarray(rgb, "RGB")
#     im.show()

#     rgb = np.zeros((1024, 1024, 3), 'uint8')
#     rgb[..., 0] = np.clip(
#         np.power(1.5 * p["image"][:, :, 0], 1.0 / 2.2) * 255, 0, 255)
#     rgb[..., 1] = np.clip(
#         np.power(1.5 * p["image"][:, :, 1], 1.0 / 2.2) * 255, 0, 255)
#     rgb[..., 2] = np.clip(
#         np.power(1.5 * p["image"][:, :, 2], 1.0 / 2.2) * 255, 0, 255)

#     im = Image.fromarray(rgb, "RGB")
#     im.show()
