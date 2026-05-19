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
python3 kissingfugal-sparse-acm-dblp.py (for the acm-dblp dataset)
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


