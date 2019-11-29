import inception_blocks_v2
import fr_utils
import facenet
import pickle
import os

def dump(database, n):
    arr = [arr for arr in range(1, 5)]
    if n not in arr:
        return None
    str = ' '
    if n == 1:
       str = '\Billdata.pickle'

    if n == 2:
        str = '\Elondata.pickle'

    if n == 3:
        str = '\Stevedata.pickle'

    if n == 4:
        str = '\BillElonStevedata.pickle'
    with open(os.getcwd() + '\dump'+ str, 'wb') as f1:
        pickle.dump(database, f1)
    pass

if __name__ == "__main__":
    n = 4
    database = facenet.prepare_database(n)
    print(database)
    dump(database,n)
