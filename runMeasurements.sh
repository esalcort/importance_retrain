#!/bin/bash

RETRAIN_SIZE=(1 2 3 4 5 6 7 8 9)

RETRAIN_RATES=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1)

METHOD="loss gnorm uniform"

for method in $METHOD
do

   echo $method

for size in "${RETRAIN_SIZE[@]}"
do
   PRETRAIN=$(((10-$size) * 10))

   echo $PRETRAIN

for rate in "${RETRAIN_RATES[@]}"
do

   echo $rate  
    VAR=$rate 
   python3 importance_retrain.py rate_retrain MNIST $method --retrain_size 0.$size --retrain_rate $rate --load_model mnist_uniform_$PRETRAIN.h5 >> results.log

done   
done
done

#   python3 importance_retrain.py rate_retrain MNIST gnorm --retrain_size 0.$size --retrain_rate $rate --load_model mnist_uniform_$PRETRAIN.h5 &> /dev/null



#  python3 importance_retrain.py rate_retrain MNIST uniform --retrain_size 0.$size --retrain_rate $rate --load_model mnist_uniform_$PRETRAIN.h5 &> /dev/null
