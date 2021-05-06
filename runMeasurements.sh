#!/bin/bash

RETRAIN_SIZE=(4 6)

RETRAIN_RATES=(0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5)

METHOD="loss gnorm uniform"
#METHOD="uniform"

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
    #python3 importance_retrain.py rate_retrain MNIST $method --retrain_size 0.$size --retrain_rate $rate --load_model mnist_uniform_$PRETRAIN.h5 >> results.log 2> /dev/null
    
    python3 importance_retrain.py rate_retrain CIFAR-10 $method --retrain_size 0.$size --retrain_rate $rate --load_model cifar10_uniform_$PRETRAIN.h5 >> results.log 2> /dev/null
  

done   
done
done

#   python3 importance_retrain.py rate_retrain MNIST gnorm --retrain_size 0.$size --retrain_rate $rate --load_model mnist_uniform_$PRETRAIN.h5 &> /dev/null



#  python3 importance_retrain.py rate_retrain MNIST uniform --retrain_size 0.$size --retrain_rate $rate --load_model mnist_uniform_$PRETRAIN.h5 &> /dev/null
