import os

import psycopg2

exit = True
connection = None

try:
    print("Скачивание данных с сервер.")
    sql = " "
    n = int(input(
        "Введите 1 для скачивание данных для Bill_Gates, 2 - Elon_Musk, 3 - Steve_Jobs, 4 - Bill_Elon_Steve, 5 - FaceRecoModel, 6 - weights_dict: "))
    if n == 1:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\dump\Billdata1.pickle') from datafacenet where name = 'Billdata';"

    elif n == 2:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\dump\Elondata1.pickle') from datafacenet where name = 'Elondata';"

    elif n == 3:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\dump\Stevedata1.pickle') from datafacenet where name = 'Stevedata';"

    elif n == 4:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\dump\BillElonStevedata1.pickle') from datafacenet where name = 'BillElonStevedata';"

    elif n == 5:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\model\FaceRecoModel1.h5') from datafacenet where name = 'FaceRecoModeldata';"

    elif n == 6:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\dump\weights_dict1.pickle') from datafacenet where name = 'weights_dict_data';"

    else:
        print("Ошибка выбора варианта!")

    connection = psycopg2.connect(dbname='facenet', user='postgres', password='1234', host='127.0.0.1')
    with connection.cursor() as cursor:
        cursor.execute(sql)
        connection.commit()
        cursor.close()


except psycopg2.OperationalError:
    print("Ошибка соединения с базой данных!")
    exit = False

finally:
    if exit == True:
        connection.close()
        print("Соединение закрыто")
