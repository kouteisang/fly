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
# chunk fast with 2 pass
python3 kissing-fast-without2pass.py(little bit faster than kissing-fast but uses a little bit more memory due to missing 2 passes)
python3 kissing-fast.py

# Hyperparameters 



**m**: the size of the low rank matrix (n*m), for the fast chunk vervion, m is choosen automaticaly and for others m = n / 10 works very well.

**chunk size**: the size of each chunk.

**beta**: Bata control the probability distribution of the softmax, for now, 10 works well (25 works better for even larger datsets, waiting for ablation studies).

**col_penalty**: control the loss penality of whether each column equals to 1, for now, 200 works well.

**learning_rate**: learning rate for now 0.01 works well.

**mu**: To control the feature loss, please check the fugal paper for this parameter setting

**max_iteration**: The number of epochs, for now, we set it to 10000, but please try early stopping or 
some other techiniques, to save the time(3000 is sufficient for even large datasets, 3500 to be safe).

**Gradient descent algorithm**: For now, I use Adam, but please try others such as SGD.

