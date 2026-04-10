import networkx as nx
import time
import numpy as np
from helpers.pred import fly
from helpers.metrics import eval_align

if __name__ == "__main__":
    n = 1004
    k = n//2
    # 加载图数据
    query = open('/home/cheng/fly/data/real_noise/MultiMagna/yeast0_Y2H1.txt', 'r')
    lines = query.readlines()
    Gquery = nx.Graph()
    for i in range(n): Gquery.add_node(i)
    for line in lines:
        u_v = (line[:-1].split(' '))
        u = int(u_v[0])
        v = int(u_v[1])
        Gquery.add_edge(u, v)

    # target file
    target = open("/home/cheng/fly/data/real_noise/MultiMagna/yeast25_Y2H1.txt", "r")
    lines = target.readlines()
    Gtarget = nx.Graph()
    for i in range(n): Gtarget.add_node(i)
    for line in lines:
        u_v = (line[:-1].split(' '))
        u = int(u_v[0])
        v = int(u_v[1])
        Gtarget.add_edge(u, v)

    
    gmb = np.arange(Gtarget.number_of_nodes())  # ground truth: identity mapping

    # 运行FUGAL
    time_start = time.time()
    ans = fly(Gquery, Gtarget, n, k,  mu=0.5, niter=15)
    time_end = time.time()

    print("Time taken: ", time_end - time_start)
    
    ma = np.array([pair[0] for pair in ans])  # query
    mb = np.array([pair[1] for pair in ans])  # target


    gacc, acc, _ = eval_align(ma, mb, gmb)
    print("gacc: ", gacc)
    print("acc: ", acc)
    