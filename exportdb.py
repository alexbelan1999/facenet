import psycopg2
import os

exit = True
connection = None
n = 4
try:
    print("Тестирование PostgreSQL")
    sql = " "
    if n == 1:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\pickle\Billdata.pickle') from datafacenet where name = 'Billdata';"

    if n == 2:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\pickle\Elondata.pickle') from datafacenet where name = 'Elondata';"

    if n == 3:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\pickle\Stevedata.pickle') from datafacenet where name = 'Stevedata';"

    if n == 4:
        sql = "SELECT lo_export(object,'" + os.getcwd() + "\pickle\BillElonStevedata.pickle') from datafacenet where name = 'BillElonStevedata';"

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