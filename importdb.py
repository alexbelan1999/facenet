import psycopg2
import os

exit = True
connection = None
n = 4
try:
    print("Тестирование PostgreSQL")
    sql = " "
    if n == 1:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('Billdata', lo_import('"+ os.getcwd() + '\dump\Billdata.pickle' +"'));"

    if n == 2:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('Elondata', lo_import('" + os.getcwd() + '\dump\Elondata.pickle' + "'));"

    if n == 3:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('Stevedata', lo_import('" + os.getcwd() + '\dump\Stevedata.pickle' + "'));"

    if n == 4:
        sql = "INSERT INTO public.datafacenet (name, object) VALUES ('BillElonStevedata', lo_import('" + os.getcwd() + '\dump\BillElonStevedata.pickle' + "'));"

    connection = psycopg2.connect(dbname='facenet',user='postgres', password='1234',host='127.0.0.1')
    with connection.cursor() as cursor:

        cursor.execute(sql)
        connection.commit()
        cursor.close()


except psycopg2.OperationalError:
    print("Ошибка соединения с базой данных!")
    exit = False

finally:
    if(exit==True):
        connection.close()
        print("Соединение закрыто")