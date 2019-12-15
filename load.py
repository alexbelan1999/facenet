import os
import pickle


def load(n):
    database = {}

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

    with open(os.getcwd() + '\dump' + str, 'rb') as f:
        database = pickle.load(f)
    return database


if __name__ == "__main__":
    database = {}
    n = int(
        input("Введите 1 - Billdata.pickle, 2 - Elondata.pickle, 3 - Stevedata.pickle, 4 - BillElonStevedata.pickle: "))
    database = load(n)
