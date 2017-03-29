This is a parallel SESOP implementation in TensorFlow.

To start working we need to setup our enviorment:
-------------------------------------------------

0. (Only the very first time) install profile in anaconda:
conda create -n tensorflow

1. First we need to activate tensorflow profile in anaconda:
source activate tensorflow

2. Open jupyther notebook

3. if CG doesnt work (using scipy interface),
edit file /home/shai/anaconda/lib/python2.7/site-packages/tensorflow/contrib/opt/python/training/external_optimizer.py


Where stuff are?
----------------
/home/shai/anaconda/envs/tensorflow
/home/shai/anaconda/lib/python2.7/site-packages/tensorflow
/home/shai/anaconda/lib/python2.7/site-packages/tensorflow/include/tensorflow

Libaries:
---------

/usr/local/cuda-8.0/extras/CUPTI/lib64/libcupti.so.8.0
/usr/local/cuda-8.0/extras/CUPTI/lib64/libcupti.so.8.0.61
shai@gpu-plx01:~/tensorflow/parallel_sesop$ 
shai@gpu-plx01:~/tensorflow/parallel_sesop$ 
shai@gpu-plx01:~/tensorflow/parallel_sesop$ echo $LD_LIBRARY_PATH 

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/extras/CUPTI/lib64/:/usr/local/cuda-8.0/lib64:/usr/lib64:/usr/lib


