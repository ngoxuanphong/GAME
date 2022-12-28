import mysql.connector

mydb = mysql.connector.connect(
  host="125.212.218.28",
  user="iqnrecqv_phong",
  password="Phong12345",
  database="iqnrecqv_v"
)

mycursor = mydb.cursor()

# mycursor.execute("SELECT * FROM CodeAgent")

# myresult = mycursor.fetchall()

# for x in myresult:
#   print(x)

# mycursor = mydb.cursor()

# sql = "INSERT INTO CodeAgent (CodeID, UserID, SystemVISID, Name) VALUES (%s, %s, %s, %s)"
# val = (1, 12345, 1, 'Agent')
# mycursor.execute(sql, val)

# mydb.commit()

# print(mycursor.rowcount, "record inserted.")