import facenet
import pickle

import facenet


def dump(database, n):
    str = ' '
    if n == 1:
        str = '\Billdata.pickle'

    elif n == 2:
        str = '\Elondata.pickle'

    elif n == 3:
        str = '\Stevedata.pickle'

    elif n == 4:
        str = '\BillElonStevedata.pickle'

    else:
        print("Ошибка выбора варианта!")
        return None

    with open(os.getcwd() + '\dump' + str, 'wb') as f1:
        pickle.dump(database, f1)
    pass


if __name__ == "__main__":
    n = int(
        input("Введите 1 - Billdata.pickle, 2 - Elondata.pickle, 3 - Stevedata.pickle, 4 - BillElonStevedata.pickle: "))
    database = facenet.prepare_database(n)
    dump(database, n)
