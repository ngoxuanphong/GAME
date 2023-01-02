from setup import SHOT_PATH, DRIVE_FOLDER
from server.mysql_connector import get_db_cursor
import os, time, shutil
import pandas as pd
import numpy as np
import zipfile

def copy_agent_bool(agent_name):
    bool_copy_file = False
    os.mkdir(f'{SHOT_PATH}Agent/{agent_name}')
    shutil.copy2(f'{DRIVE_FOLDER}CodeAgent/{agent_name}.py', f'{SHOT_PATH}Agent/{agent_name}/Agent_player.py')
    time.sleep(3)
    if os.path.exists(f'{SHOT_PATH}Agent/{agent_name}/Agent_player.py'):
        bool_copy_file = True
    return bool_copy_file

def check_sleep_copt_agent(agent_name):
    if os.path.exists(f'{DRIVE_FOLDER}CodeAgent/{agent_name}.py'):
        bool_copy_file = copy_agent_bool(agent_name)
    else:
        time.sleep(30) #Thời gian để load lại drive
        bool_copy_file = copy_agent_bool(agent_name)
    return bool_copy_file


def copy_new_agent():
    mycursor, mydb = get_db_cursor()
    mycursor.execute(f"SELECT * FROM CodeBot WHERE NotificateID = '107'")
    myresult = mycursor.fetchall()

    if len(myresult) > 0: #Nếu có file mới vừa được cập nhật ở drive
        #Copy về drive, cập nhật trạng thái thành waiting(100)

        info_get = myresult[0]

        id_from_table = info_get[0]
        user_id = info_get[1]
        agent_name = info_get[2]

        system_vis_id = 1
        Notificate_ID = 100

        bool_copy_file = check_sleep_copt_agent(agent_name)

        if bool_copy_file:

            val = (system_vis_id, agent_name, Notificate_ID, id_from_table, time.strftime('%Y-%m-%d %H:%M:%S'))
            sql = f"INSERT INTO HistoryCodeAgent(SystemVISID, Name, NotificateID, BotID, CreateOn) VALUES (%s, %s, %s, %s, %s)"
            mycursor.execute(sql, val)

            val = (Notificate_ID, agent_name)
            sql = f"UPDATE CodeBot SET NotificateID = %s WHERE CodeID = %s"
            mycursor.execute(sql, val)

            mycursor.execute("SELECT * FROM CodeBot")
            # print(mycursor.fetchall())

            df_agent = pd.read_json(f'{SHOT_PATH}Log/StateAgent.json')
            print(df_agent)
            df_agent.loc[len(df_agent)] = [agent_name, np.nan, np.nan, np.nan]
            print('Agent name', agent_name, len(df_agent))
            df_agent.to_json(f'{SHOT_PATH}Log/StateAgent.json')
            print(df_agent)

            mydb.commit()



def get_notifi_server(type_code, msg, name_type, *args):
    mycursor, mydb = get_db_cursor()
    NotifiID = None
    if type_code == 'Agent':
        if msg == 'WAITING': NotifiID = 100
        if msg == 'CHECKING': NotifiID = 101
        if msg == 'NOBUG': NotifiID = 102
        if msg == 'BUG': NotifiID = 103
        if msg == 'TRAINING': NotifiID = 104
        if msg == 'TESTING': NotifiID = 105
        if msg == 'FINISHED': NotifiID = 106
        if msg == 'UPLOAD SUCCESSFUL': NotifiID = 107

    if type_code == 'Env':
        if msg == 'WAITING': NotifiID = 200
        if msg == 'CHECKING': NotifiID = 201
        if msg == 'NOBUG': NotifiID = 202
        if msg == 'BUG': NotifiID = 203
        if msg == 'TRAINING': NotifiID = 204
        if msg == 'TESTING': NotifiID = 205
        if msg == 'FINISHED': NotifiID = 206  
        if msg == 'UPLOAD SUCCESSFUL': NotifiID = 207

    if NotifiID != None:
        sql = f"UPDATE CodeBot SET NotificateID = %s WHERE CodeID = %s"
        val = (NotifiID, name_type) 
        mycursor.execute(sql, val)

        sql = f"UPDATE CodeBot SET CreateOn = %s WHERE CodeID = %s"
        val = (time.strftime('%Y-%m-%d %H:%M:%S'), name_type) 
        mycursor.execute(sql, val)

        if len(args) > 0:
            sql = f"UPDATE CodeBot SET ScoreElo = %s WHERE CodeID = %s"
            val = (args[0], name_type)
            mycursor.execute(sql, val)
        
        mydb.commit()


def update_notificate_by_id(ID, msg, *args):
    mycursor, mydb = get_db_cursor()
    sql = f"UPDATE HistorySystem SET NotificateID = %s WHERE ID = %s"
    if msg == 'ERROR FORMAT':val = (120, ID)
    if msg == 'CANT UNZIP':val = (118, ID)
    if msg == 'WAITING': val = (110, ID)
        
    if msg == 'CHECKING': val = (111, ID)
    if msg == 'NO BUG': val = (112, ID)
    if msg == 'BUG': val = (113, ID)

    mycursor.execute(sql, val)
    mycursor.execute("SELECT * FROM HistorySystem")
    print(mycursor.fetchall())
    mydb.commit()


def copy_new_env():
    mycursor, mydb = get_db_cursor()
    mycursor.execute(f''' SELECT hs.ID, su.Name
                    FROM  HistorySystem hs
                    Left join auth_user au on au.id = hs.UserID
                    Left join SystemUser su on su.SystemID = hs.SystemID
                    Left join Notificate n on n.NotificateID = hs.NotificateID
                    Where hs.NotificateID = 117''')

    data_env_upload = mycursor.fetchall()
    if len(data_env_upload) > 0:
        ID = data_env_upload[0][0]
        env_name = data_env_upload[0][1]

        path_to_zip_file = f'{DRIVE_FOLDER}/System/{env_name}.zip'
        directory_to_extract_to = f'{SHOT_PATH}Base/{env_name}'
    
        def upzip_env(ID):
            try:
                if os.path.exists(path_to_zip_file) == False:
                    time.sleep(30)
                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(directory_to_extract_to)
                    time.sleep(2)
                    if os.path.exists(f'{directory_to_extract_to}/env.py') == False or os.path.exists(f'{directory_to_extract_to}/README.md') == False:
                        print(f'{directory_to_extract_to}/env.py')
                        print(f'{directory_to_extract_to}/README.md')
                        update_notificate_by_id(ID, 'ERROR FORMAT')
            except:
                update_notificate_by_id(ID, 'CANT UNZIP')

        upzip_env(ID)
        if os.path.exists(f'{directory_to_extract_to}/env.py'):
            update_notificate_by_id(ID, 'WAITING')

            df_env = pd.read_json(f'{SHOT_PATH}Log/StateEnv.json')
            df_env.loc[len(df_env)] = [env_name, np.nan, np.nan, np.nan]
            df_env.to_json(f'{SHOT_PATH}Log/StateEnv.json')
    
# copy_new_env()

