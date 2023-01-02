import mysql.connector

def get_db_cursor():
  mydb = mysql.connector.connect(
    host="125.212.218.28",
    user="iqnrecqv_phong",
    password="Phong12345",
    database="iqnrecqv_v"
  )

  mycursor = mydb.cursor(buffered=True)
  return mycursor, mydb

