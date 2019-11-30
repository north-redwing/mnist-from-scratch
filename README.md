
# mnist-from-scratch

NumPy implementation of a begginner classification task on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset with a 3 Layers Neural Network model.  
I tested this experiments for understanding black box of deep neural networks.  
This is based on [oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch).

## Usage
### MNIST download
In `~/mnist-from-scratch/data/ubyte/` ,  
if you use macOS, Execute the script,

```
$ curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
$ curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
$ curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
$ curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```


if you use Linux, Execute the script,
```
$ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```
then you will get MNIST dataset as byte extension.


### Convert to NumPy array
Execute the script,
```
python ~/mnist-from-scratch/src/make_arr.py
```

### Learning
Execute the script,
```
$ python3 ~/mnist-from-scratch/src/learning.py sample 4000 0.1 100 0 -debug -save
```

Here is keywords arguments and optional arguments.  
Note:  
You will train on MNIST dataset with noise from 1% to noise_per_max, every 5%.  
If noise_per_max is nonzero, then you will train on MNIST dataset without noise.

```
keywords arguments:
  filename             path to save data
  iters_num            number of iterations
  lr                   learning rate
  hidden_size_list     list of number of hidden neuron
  noiose_per_max       max noise percentage.
  
optional arguments:
  -debug print         debug mode
  -save                save parameters, graphs, arrays
 ```


## Results
- Accuracy
![sample_noise0%_acc_graph.jpg](https://qiita-image-store.s3.amazonaws.com/0/324488/8a74dd11-1a49-7066-4898-322ca8b04d45.jpeg)

- Loss
![sample_noise0%_loss_graph.jpg](https://qiita-image-store.s3.amazonaws.com/0/324488/aa916fd3-998f-ccb6-1b22-321055853555.jpeg)

- Confusion Matrix
![sample_noise0%_cm_graph.jpg](https://qiita-image-store.s3.amazonaws.com/0/324488/b32c62de-1c2b-376d-af21-65bd347a2fae.jpeg)

- Accuracy with variable noise percentage
![sample_compare.jpg](https://qiita-image-store.s3.amazonaws.com/0/324488/53ce5869-b738-b4d9-fc40-78f528bf0896.jpeg)

- Visualization of W1
![W1.png](https://qiita-image-store.s3.amazonaws.com/0/324488/04889792-3d51-7662-5ffc-2cafb3004352.png)

- Visualization of how W1 affects each number
![W2.png](https://qiita-image-store.s3.amazonaws.com/0/324488/85fe5a18-5b96-f6d4-e2b8-21399326eeef.png)
