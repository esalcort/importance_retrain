import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from blinker import signal
import numpy as np
import argparse
import time
import os
import sys
sys.path.append('importance-sampling')
sys.path.append('importance-sampling/examples')

from importance_sampling.training import ImportanceTraining
from importance_sampling.models import wide_resnet
from importance_sampling.datasets import CIFAR10, CIFAR100, ZCAWhitening
from examples.cifar10_resnet import TrainingSchedule
from importance_sampling.layers.normalization import LayerNormalization

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment',       choices=['base', 'rate_retrain', 'threshold_retrain'])
    parser.add_argument('dataset',          choices=['MNIST', 'CIFAR-10', 'CIFAR-100'])
    parser.add_argument('train_score',      choices=['loss', 'gnorm', 'uniform'])
    parser.add_argument('--random_retrain', action='store_true')
    parser.add_argument('--retrain_size',   type=float,   default=0.4)
    parser.add_argument('--batch_size',     type=int,     default=128)
    parser.add_argument('--presample',      type=float,   default=5.0)
    parser.add_argument('--epochs',         type=int,     default=10)
    parser.add_argument('--retrain_epochs',   type=int,     default=1)
    parser.add_argument('--retrain_rate',     type=float)
    parser.add_argument('--retrain_threshold',type=float)
    parser.add_argument('--load_model')
    parser.add_argument('--save_model')
    parser.add_argument('--whitening',      action='store_true')
    parser.add_argument('--augment_data',      action='store_true')
    parser.add_argument('--continue_training',  action='store_true')

    return parser

def get_dataset(dataset_name, whitening=False):
    if dataset_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    elif dataset_name == 'CIFAR-10': 
        if whitening:
            dset = ZCAWhitening(CIFAR10())
        else:
            dset = CIFAR10()
        x_train, y_train = dset.train_data[:]
        x_test, y_test = dset.test_data[:]
    elif dataset_name == 'CIFAR-100': 
        if whitening:
            dset = ZCAWhitening(CIFAR100())
        else:
            dset = CIFAR100()
        x_train, y_train = dset.train_data[:]
        x_test, y_test = dset.test_data[:]
    else:
        raise Exception('Unknown data set name %s'%(dataset_name))
    return x_train, y_train, x_test, y_test

def cifar_step_decay(epoch, lr):
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.02
    return 0.004

def load_cifar10_batch_data(start, count):
    from keras.utils.data_utils import get_file
    from keras.datasets.cifar import load_batch
    from keras.utils import np_utils

    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=
      '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

    num_batch_samples = 10000
    first_batch = 1 + int(start / num_batch_samples)
    last_batch = 1 + int((start + count - 1) / num_batch_samples)

    temp_x = np.empty((num_batch_samples, 3, 32, 32), dtype='uint8')
    temp_y = np.empty((num_batch_samples,), dtype='uint8')

    x_train = np.empty((count, 3, 32, 32), dtype='uint8')
    y_train = np.empty((count,), dtype='uint8')

    idx_start = start % 10000
    idx_end = 10000
    data_offset = 0
    rows = idx_end - idx_start

    for i in range(first_batch, last_batch+1):
        fpath = os.path.join(path, 'data_batch_'+str(i))
        (temp_x, temp_y) = load_batch(fpath)
        x_train[data_offset:data_offset+rows] = temp_x[idx_start: idx_end]
        y_train[data_offset:data_offset+rows] = temp_y[idx_start: idx_end]
        data_offset += rows
        idx_start = 0
        rows = 10000
        idx_end = 10000
        if (i + 1) == last_batch:
            rows = count - data_offset
            idx_end = rows



    # fpath = os.path.join(path, 'test_batch')
    # x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    # y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        # x_test = x_test.transpose(0, 2, 3, 1)

    # x_test = x_test.astype(x_train.dtype)
    # y_test = y_test.astype(y_train.dtype)

    y_train = np_utils.to_categorical(y_train)

    return (x_train, y_train)


def get_dataset_model(dataset_name):
    training_schedule = None
    if dataset_name == 'MNIST':
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-5),
          input_shape=(784,)))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-5)))
        model.add(Dense(10, kernel_regularizer=l2(1e-5)))
        model.add(Activation('softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy']
        )
    elif dataset_name == 'CIFAR-10': 
        training_schedule = TrainingSchedule(3 * 3600)
        model = wide_resnet(28, 2)((32, 32, 3), 10)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=SGD(lr=training_schedule.lr, momentum=0.9),
            metrics=["accuracy"]
        )
    elif dataset_name == 'CIFAR-100': 
        model = wide_resnet(28, 2)((32, 32, 3), 100)
        training_schedule = TrainingSchedule(3 * 3600)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=SGD(lr=training_schedule.lr, momentum=0.9),
            metrics=["accuracy"]
        )
    else:
        raise Exception('Unknown data set name %s'%(dataset_name))
    return model, training_schedule

def main():
    parser = get_parser()
    args = parser.parse_args()
    is_cifar = 'CIFAR' in args.dataset

    # Parser checks
    print('Parser checks...')
    assert args.retrain_size < 1 and args.retrain_size >= 0, 'Provide retrain size as a fraction of train data'
    if args.experiment == 'rate_retrain':
        assert args.retrain_rate, 'retrain_rate is required'
    elif args.experiment == 'threshold_retrain':
        assert args.retrain_threshold, 'retrain_threshold is required'

    # Get Data
    print('Get Data...')
    if args.dataset == 'CIFAR-10':
        x_retrain, y_retrain = load_cifar10_batch_data(int(50000*(1-args.retrain_size)), int(50000*args.retrain_size))
    else:
        x_train, y_train, x_test, y_test = get_dataset(args.dataset, args.whitening)
        # Partition train and retrain
        retrain_idx = len(x_train) - int(len(x_train) * args.retrain_size)
        x_retrain, y_retrain = x_train[retrain_idx:], y_train[retrain_idx:]
        x_train, y_train = x_train[:retrain_idx], y_train[:retrain_idx]
    # Get model
    if args.load_model:
        model = load_model(os.path.join('pre_trained', args.load_model), custom_objects={'LayerNormalization' : LayerNormalization})
        for layer in model.layers:
            if 'input' in layer.name:
                layer.name='orig_' + layer.name
    else:
        model, training_schedule = get_dataset_model(args.dataset)

    if args.train_score != 'uniform':
        wrapped = ImportanceTraining(
            model,
            args.presample,
            score=args.train_score
        )
    else:
        wrapped = model


    results = {}
    
    # Re-train
    print('Re-train...')
    fwd_sample_count = int(args.batch_size * args.presample)
    bwd_sample_count = int(args.retrain_rate * fwd_sample_count)
    assert bwd_sample_count >= 1, 'Select a larger retrain_rate'
    results['fwd_sample_count'] = fwd_sample_count
    results['bwd_sample_count'] = bwd_sample_count
    if args.experiment == 'base':
        retrain_time = time.perf_counter()
        history = wrapped.fit(
            x_retrain, y_retrain,
            batch_size=args.batch_size,
            epochs=args.retrain_epochs,
            verbose=0
        )
        retrain_time = time.perf_counter() - retrain_time
    elif args.experiment == 'rate_retrain':
        if args.train_score == 'uniform':
            retrain_time = time.perf_counter()
            for b in range(0, len(y_retrain), fwd_sample_count):
                fwd_x, fwd_y = x_retrain[b:b+fwd_sample_count], y_retrain[b:b+fwd_sample_count]
                # Simulate Forward pass:
                model.evaluate(fwd_x, fwd_y, verbose=0)
                if bwd_sample_count < len(fwd_x):
                    # Sample uniformly
                    sample_idx = np.random.choice(len(fwd_x), bwd_sample_count, replace=False)
                    # Simulate Backward pass:
                    model.fit(fwd_x[sample_idx], fwd_y[sample_idx], batch_size=bwd_sample_count, verbose=0,
                        epochs=args.retrain_epochs)
                else:
                    model.fit(fwd_x, fwd_y, batch_size=len(fwd_x), verbose=0, epochs=args.retrain_epochs)
            retrain_time = time.perf_counter() - retrain_time
            
        else:
            scores_list = list()
            def on_evaluate(metrics):
                scores_list.append(metrics[3])
            signal("is.evaluate_batch").connect(on_evaluate)
            retrain_time = time.perf_counter()
            for b in range(0, len(y_retrain), fwd_sample_count):
                fwd_x, fwd_y = x_retrain[b:b+fwd_sample_count], y_retrain[b:b+fwd_sample_count]
                # Simulate Forward pass
                scores_list = []
                wrapped.model.evaluate(fwd_x, fwd_y)
                # Collect importance metrics
                scores = np.concatenate(scores_list).flatten() + 10**-9
                p = scores / scores.sum()
                if bwd_sample_count < len(fwd_x):
                    # Sample based on metrics
                    sample_idx = np.random.choice(len(fwd_x), bwd_sample_count, replace=False, p=p)
                    # Simulate Backward pass
                    model.fit(fwd_x[sample_idx], fwd_y[sample_idx], batch_size=bwd_sample_count, verbose=0,
                        epochs=args.retrain_epochs)
                else:
                    model.fit(fwd_x, fwd_y, batch_size=len(fwd_x), verbose=0, epochs=args.retrain_epochs)
            retrain_time = time.perf_counter() - retrain_time
    elif args.experiment == 'threshold_retrain':
        scores_list = list()
        def on_evaluate(metrics):
            scores_list.append(metrics[3])
        signal("is.evaluate_batch").connect(on_evaluate)
        retrain_time = time.perf_counter()
        wrapped.model.evaluate(x_retrain, y_retrain)
        scores = np.concatenate(scores_list).flatten()
        samples_mask = scores > args.retrain_threshold
        model.fit(
            x_retrain[samples_mask], y_retrain[samples_mask],
            batch_size=args.batch_size,
            epochs=args.retrain_epochs,
            verbose=0
        )
        retrain_time = time.perf_counter() - retrain_time
        results['th_true_count'] = samples_mask.sum()

    results['retrain_time'] = retrain_time

    # Evaluate
    print('Evaluate...')
    if args.train_score == 'uniform':
        fwd_time_per_sample = time.perf_counter()
        score = model.evaluate(x_retrain[:fwd_sample_count], y_retrain[:fwd_sample_count], verbose=0)
        fwd_time_per_sample = (time.perf_counter() - fwd_time_per_sample) / fwd_sample_count
    else:
        fwd_time_per_sample = time.perf_counter()
        score = wrapped.model.evaluate(x_retrain[:fwd_sample_count], y_retrain[:fwd_sample_count])
        fwd_time_per_sample = (time.perf_counter() - fwd_time_per_sample) / fwd_sample_count
    results['fwd_time_per_sample'] = fwd_time_per_sample

    # results['retrain_loss'] = score[0]
    # results['retrain_acc'] = score[1]

    # results['train_loss'], results['train_acc'] = model.evaluate(x_train, y_train, verbose=0)
    # results['retrain_loss'], results['retrain_acc'] = model.evaluate(x_retrain, y_retrain, verbose=0)

    # Print results
    print('Configuration:')
    print(args)
    print('\nResults:')
    print(results)





if __name__ == "__main__":
    main()
