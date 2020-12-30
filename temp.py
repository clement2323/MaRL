import multiprocessing as mp

def power_n(x, n, y):
    return x ** n + y

if __name__ == '__main__':    
    pool = mp.Pool(8)
    result = pool.starmap(power_n, [(x, 2) for x in range(20)])
    print(result)
