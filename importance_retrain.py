import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2

import argparse
import time
import sys
sys.path.append('importance-sampling')

from importance_sampling.training import ImportanceTraining
from importance_sampling.models import wide_resnet
from importance_sampling.datasets import CIFAR10, ZCAWhitening

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment',       choices=['base', 'rate_retrain', 'threshold_retrain'])
    parser.add_argument('dataset',          choices=['MNIST', 'CIFAR-10'])
    parser.add_argument('train_score',      choices=['loss', 'gnorm', 'uniform'])
    parser.add_argument('--random_retrain', action='store_true')
    parser.add_argument('--retrain_size',   type=float,   default=0.4)
    parser.add_argument('--batch_size',     type=int,     default=128)
    parser.add_argument('--presample',      type=float,   default=5.0)
    parser.add_argument('--epochs',         type=int,     default=10)
    parser.add_argument('--retrain_epochs',   type=int,     default=1)
    parser.add_argument('--retrain_rate',     type=float)
    parser.add_argument('--retrain_threshold',type=float)

    return parser


def get_dataset(dataset_name):
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
        dset = ZCAWhitening(CIFAR10())
        x_train, y_train = dset.train_data[:]
        x_test, y_test = dset.test_data[:]
    else:
        raise Exception('Unknown data set name %s'%(dataset_name))
    return x_train, y_train, x_test, y_test

def get_dataset_model(dataset_name):
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
        model = wide_resnet(28, 2)((32, 32, 3), 10)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=["accuracy"]
        )
    else:
        raise Exception('Unknown data set name %s'%(dataset_name))
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Parser checks
    assert args.retrain_size < 1 and args.retrain_size >= 0, 'Provide retrain size as a fraction of train data'

    # Get Data
    x_train, y_train, x_test, y_test = get_dataset(args.dataset)
    # Partition train and retrain
    if args.retrain_size > 0:
        retrain_idx = len(x_train) - int(len(x_train) * args.retrain_size)
        x_retrain, y_retrain = x_train[retrain_idx:], y_train[retrain_idx:]
        x_train, y_train = x_train[:retrain_idx], y_train[:retrain_idx]
    else:
        x_retrain, y_retrain = [], []
    # Get model
    model = get_dataset_model(args.dataset)

    if args.train_score != 'uniform':
        wrapped = ImportanceTraining(
            model,
            args.presample,
            score=args.train_score
        )
    else:
        wrapped = model


    results = {}
    # Train
    train_time = time.time()
    history = wrapped.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0
    )
    train_time = time.time() - train_time
    results['train_time'] = train_time

    # Re-train
    if args.experiment == 'base':
        retrain_time = time.time()
        history = wrapped.fit(
            x_retrain, y_retrain,
            batch_size=args.batch_size,
            epochs=args.retrain_epochs,
            verbose=0
        )
        retrain_time = time.time() - retrain_time
    results['retrain_time'] = retrain_time

    # Evaluate
    test_batch_count = int(len(x_test)/args.batch_size)
    if args.train_score == 'uniform':
        fwd_time_per_batch = time.time()
        score = model.evaluate(x_test, y_test, verbose=0)
        fwd_time_per_batch = (time.time() - fwd_time_per_batch) / test_batch_count
    else:
        fwd_time_per_batch = time.time()
        score = wrapped.model.evaluate(x_test, y_test)
        fwd_time_per_batch = (time.time() - fwd_time_per_batch) / test_batch_count
    results['fwd_time_per_batch'] = fwd_time_per_batch

    results['test_loss'] = score[0]
    results['test_acc'] = score[1]

    results['train_loss'], results['train_acc'] = model.evaluate(x_train, y_train, verbose=0)
    results['retrain_loss'], results['retrain_acc'] = model.evaluate(x_retrain, y_retrain, verbose=0)

    # Print results
    print('Configuration:')
    print(args)
    print('\nResults:')
    print(results)





if __name__ == "__main__":
    main()