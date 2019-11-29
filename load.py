import pickle
import os

def load(n):
    database = {}
    arr = [arr for arr in range(1, 5)]
    if n not in arr:
        return database
    str = ' '
    if n == 1:
        str = '\Billdata.pickle'

    if n == 2:
        str = '\Elondata.pickle'

    if n == 3:
        str = '\Stevedata.pickle'

    if n == 4:
        str = '\BillElonStevedata.pickle'
    with open(os.getcwd() +'\dump' + str, 'rb') as f:
        database = pickle.load(f)
    return database

if __name__ == "__main__":
    database = {}
    n = 4
    database = load(n)
    print(database)