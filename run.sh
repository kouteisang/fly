python3 main.py --mode dense --k 500 --niter 15 --mu 0.5 --device cpu
python3 main.py --mode hybrid --k 100 --niter 15 --mu 0.5 --device cpu
python3 main.py --mode sparse --k 100 --niter 15 --mu 0.5 --device cpu
python3 main.py --mode lowrank --k 100 --niter 15 --mu 0.5 --device cuda