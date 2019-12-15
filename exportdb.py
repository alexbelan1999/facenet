import os

import psycopg2

exit = True
connection = None
try:
    print("Загрузка данных на сервер.")
    sql = " "
    n = int(input(
        "Введите 1 для загрузки данных Bill_Gates, 2 - Elon_Musk, 3 - Steve_Jobs, 4 - Bill_Elon_Steve, 5 - FaceRecoModel, 6 - weights_dict: "))
    if n == 1:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('Billdata', lo_import('" + os.getcwd() + '\dump\Billdata.pickle' + "'));"

    elif n == 2:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('Elondata', lo_import('" + os.getcwd() + '\dump\Elondata.pickle' + "'));"

    elif n == 3:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('Stevedata', lo_import('" + os.getcwd() + '\dump\Stevedata.pickle' + "'));"

    elif n == 4:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('BillElonStevedata', lo_import('" + os.getcwd() + '\dump\BillElonStevedata.pickle' + "'));"

    elif n == 5:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('FaceRecoModeldata', lo_import('" + os.getcwd() + '\model\FaceRecoModel.h5' + "'));"

    elif n == 6:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('weights_dict_data', lo_import('" + os.getcwd() + '\dump\weights_dict.pickle' + "'));"

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
