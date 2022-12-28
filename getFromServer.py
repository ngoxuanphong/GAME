from server.mysql_connector import mydb, mycursor
import mysql.connector

mycursor.execute("SELECT * FROM CodeBot")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)
