from mysql_connector import mydb, mycursor

sql = 'INSERT INTO  Notificate (NotificateID, Description) VALUES (%s, %s)'
val = [
    (100, 'Agent, Waiting'),
    (101, 'Agent, Checking'),
    (102, 'Agent, NO BUG'),
    (103, 'Agent, BUG'),
    (104, 'Agent, Training'),
    (105, 'Agent, Testing'),
    (106, 'Agent, Finished'),
    (107, 'Agent, Successful upload'),

    (110, 'Environment, Waiting'),
    (111, 'Environment, Checking'),
    (112, 'Environment, NO BUG'),
    (113, 'Environment, BUG'),
    (114, 'Environment, Training'),
    (115, 'Environment, Testing'),
    (116, 'Environment, Finished'),
    (117, 'Environment, Successful upload'),
    (118, 'Environment, Can not extract zip files'),

    (120, 'Format file error')

    (400, 'Agent, Agent have bug'),
    (401, 'Agent, No Agent function'),
    (402, 'Agent, No DataAgent function'),

    (403, 'Environment, Environment can have infinite loop'),
    (404, 'Environment, No getActionSize funtion'), 
    (405, 'Environment, No getStateSize funtion'), 
    (406, 'Environment, No getAgentSize funtion'), 
    (407, 'Environment, No getReward funtion'), 
    (408, 'Environment, No getValidActions funtion'), 
    (409, 'Environment, No normal_main funtion'), 
    (410, 'Environment, No normal_main_2 funtion'), 
    (411, 'Environment, No numba_main_2 funtion'),

    (412, 'Environment, getActionSize function return incorrect output'),
    (413, 'Environment, getStateSize function return incorrect output'),
    (414, 'Environment, getAgentSize function return incorrect output'),

    (415, 'Environment, numba_main_2 function return incorrect output'),
    (416, 'Environment, normal_main_2 function return incorrect output'),
    (417, 'Environment, normal_main function return incorrect output'),

    (418, 'Environment, numba_main_2 function have bug'),
    (419, 'Environment, normal_main_2 function have bug'),
    (420, 'Environment, normal_main function have bug'),

    (421, 'Environment, numba_main_2 function incorrect number of matches ended'),
    (422, 'Environment, normal_main_2 function incorrect number of matches ended'),
    (423, 'Environment, normal_main function incorrect number of matches ended'),

    (424, 'Agent, Upload failed'),
    (425, 'Environment, Upload failed'),
]
mycursor.executemany(sql, val)


mydb.commit()
print(mycursor.rowcount)
