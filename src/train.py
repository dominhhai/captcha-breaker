from __future__ import print_function, absolute_import, division
import argparse
import numpy as np
import tensorflow as tf
from model import captcha_classifier

MODEL_LOG_DIR = 'checkpoint'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=3000, type=int,
                      help='number of training steps')

def main(argv):
    # run argv
    args = parser.parse_args(argv[1:])
    print(args. batch_size, args.train_steps)
    # Load training and eval data
    with np.load('data/captcha.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    # Set up logging for predictions
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_train},
        y=y_train,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True
    )
    captcha_classifier.train(
        input_fn=train_input_fn,
        steps=args.train_steps,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = captcha_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
