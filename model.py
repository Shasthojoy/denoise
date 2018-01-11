import tensorflow as tf


def model_fn(features, labels, mode, params):
    x = features
    tf.summary.image("input/color", x[:, :, :, 0:3])
    tf.summary.image("input/depth", x[:, :, :, 3:4])
    tf.summary.image("input/normal", x[:, :, :, 4:7])
    tf.summary.image("input/gloss", x[:, :, :, 7:8])
    tf.summary.image("input/target", labels[:, :, :, 0:3])

    activation = tf.nn.relu
    # 128x128x11 -> 128x128x32
    x32 = tf.layers.conv2d(
        x, 32, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Encoder/Conv32")

    # ENCODER
    # 128x128x32 -> 64x64x43
    x = tf.layers.conv2d(
        x32, 43, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Encoder/Conv43")
    x43 = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[2, 2],
        padding='SAME', name="Encoder/Pool43")

    # 64x64x43 -> 32x32x57
    x = tf.layers.conv2d(
        x43, 57, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Encoder/Conv57")
    x57 = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[2, 2],
        padding='SAME', name="Encoder/Pool57")

    # 32x32x57 -> 16x16x76
    x = tf.layers.conv2d(
        x57, 76, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Encoder/Conv76")
    x76 = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[2, 2],
        padding='SAME', name="Encoder/Pool76")

    # 16x16x76 -> 8x8x101
    x = tf.layers.conv2d(
        x76, 101, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Encoder/Conv101")
    x101 = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[2, 2],
        padding='SAME', name="Encoder/Pool101")

    # 8x8x101 -> 4x4x101
    x = tf.layers.conv2d(
        x101, 101, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv101")
    dx101 = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[2, 2],
        padding='SAME', name="Decoder/Pool101")

    # DECODER
    # 4x4x101 -> 8x8x101
    two = tf.constant(2, dtype=tf.int32)
    s = tf.shape(dx101)
    nh = s[1] * two
    nw = s[2] * two
    x = tf.image.resize_nearest_neighbor(
        dx101, (nh, nw), name="Decoder/Upsample101")
    x = tf.concat([x, x101], axis=3, name="Decoder/Concat101")
    x = tf.layers.conv2d(
        x, 101, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv101_0")
    x = tf.layers.conv2d(
        x, 101, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv101_1")
    x = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[1, 1],
        padding='SAME', name="Decoder/Pool101")

    # 8x8x101 -> 16x16x76
    s = tf.shape(x)
    nh = s[1] * two
    nw = s[2] * two
    x = tf.image.resize_nearest_neighbor(
        x, (nh, nw), name="Decoder/Upsample76")
    x = tf.concat([x, x76], axis=3, name="Decoder/Concat76")
    x = tf.layers.conv2d(
        x, 76, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv76_0")
    x = tf.layers.conv2d(
        x, 76, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv76_1")
    x = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[1, 1],
        padding='SAME', name="Decoder/Pool76")

    # 16x16x76 ->  32x32x57
    s = tf.shape(x)
    nh = s[1] * two
    nw = s[2] * two
    x = tf.image.resize_nearest_neighbor(
        x, (nh, nw), name="Decoder/Upsample57")
    x = tf.concat([x, x57], axis=3, name="Decoder/Concat57")
    x = tf.layers.conv2d(
        x, 57, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv57_0")
    x = tf.layers.conv2d(
        x, 57, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv57_1")
    x = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[1, 1],
        padding='SAME', name="Decoder/Pool57")

    # 32x32x57 ->  64x64x43
    s = tf.shape(x)
    nh = s[1] * two
    nw = s[2] * two
    x = tf.image.resize_nearest_neighbor(
        x, (nh, nw), name="Decoder/Upsample43")
    x = tf.concat([x, x43], axis=3, name="Decoder/Concat43")
    x = tf.layers.conv2d(
        x, 43, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv43_0")
    x = tf.layers.conv2d(
        x, 43, kernel_size=[3, 3],
        padding='SAME', activation=activation, name="Decoder/Conv43_1")
    x = tf.layers.max_pooling2d(
        x, pool_size=[2, 2], strides=[1, 1],
        padding='SAME', name="Decoder/Pool43")

    # 64x64x43 -> 128x128x3
    s = tf.shape(x)
    nh = s[1] * two
    nw = s[2] * two
    x = tf.image.resize_nearest_neighbor(
        x, (nh, nw), name="Decoder/Upsample32")
    x = tf.concat([x, x32], axis=3, name="Decoder/Concat32")
    output = tf.layers.conv2d(
        x, 3, kernel_size=[3, 3],
        padding='SAME', name="Decoder/Conv3")

    tf.summary.image("output/color", output)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'image': output,
                'orig': features[:, :, :, 0:3]
            })

    loss = tf.losses.mean_squared_error(predictions=output, labels=labels)

    eval_metric_ops = {
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            loss=loss,
            mode=mode,
            predictions={
                'image': output,
                'target': labels,
                'loss': loss
            },
            eval_metric_ops=eval_metric_ops
        )

    logging_hook = tf.train.LoggingTensorHook({"loss": loss},
                                              every_n_iter=100)

    # loss function & optimizer
    train_op = tf.train.AdamOptimizer(params['learning_rate']) \
        .minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'image': output
        },
        loss=loss,
        train_op=train_op,
        training_hooks=[logging_hook]
    )
