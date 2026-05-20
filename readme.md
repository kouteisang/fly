# Structure of the code.

For the Fugal implementation please check the  https://github.com/idea-iitd/Fugal

# package

Remember create a conda enviroment before install the packages.

```
pip install -r requirements.txt
```

# Results and hyperparameters

All the results and hyperparameters are under Result-all


# Dense version

```
python3 kissingfugal-dense.py
python3 kissingfugal-dense-acm-dblp.py (for the acm-dblp dataset)
```

# dense version with Linear Attention

```
python3 kissingfugal-dense-LA.py
```

# dense version with chunk implementation

```
python3 kissingfugal-dense-chunked.py
python3 kissingfugal-dense-chunked-acm-dblp.py (for the acm-dblp dataset)
```

# dense version with chunk fast implementation

This version automatically choose the chunk size and embedding dimension by the algorithm itself.

```
python3 kissingfugal-dense-chunked-fast.py
python3 kissingfugal-dense-chunked-fast-acm-dblp.py (for the acm-dblp dataset)
```

# Hyperparameters 



**m**: the size of the low rank matrix (n*m), for the fast chunk vervion, m is choosen automaticaly and for others m = n / 10 works very well.

**chunk size**: the size of each chunk.

**beta**: Bata control the probability distribution of the softmax, for now, 10 works well.

**col_penalty**: control the loss penality of whether each column equals to 1, for now, 200 works well.

**learning_rate**: learning rate for now 0.01 works well.

**mu**: To control the feature loss, please check the fugal paper for this parameter setting

**max_iteration**: The number of epochs, for now, we set it to 10000, but please try early stopping or 
some other techiniques, to save the time.

**Gradient descent algorithm**: For now, I use Adam, but please try others such as SGD.


