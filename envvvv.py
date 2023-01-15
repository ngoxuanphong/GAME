
import numpy as np
from numba import njit,jit
from numba.typed import List
perData = List()
perData.append(np.array([[0.]]))

CARD = 52
cards = np.arange(1,53) # Build cards
sevens = [7,20,33,46]
idxChip = [21 ,35 ,49,63]
ACTION_SIZE= 53
AGENT_SIZE = 4
STATE_SIZE = 109
# Work function
@njit()
def initEnv():
    cards = np.arange(0,52) # Build cards`
    np.random.shuffle(cards) # Shuffle cards 
    env = np.full(67,0)
    env[0:8] = np.array([-1,7,-1,20,-1,33,-1,46]) # Khởi tạo giá trị lớn nhất + giá trị nhỏ nhất của các chất của thẻ trên bàn 

    env[8:21] = cards[0:13]   # Chia bai cho nguoi choi 0
    env[22:35] = cards[13:26] # Chia bai cho nguoi choi 1
    env[36:49] = cards[26:39] # Chia bai cho nguoi choi 2
    env[50:63] = cards[39:52] # Chia bai cho nguoi choi 3
    idx = np.array([21 ,35 ,49,63])
    env[idx] = 100  # Khoi tao chip cua moi nguoi choi
    env[64] = 0 # turn nguoi choi
    env[65] = 0 # tempStoreChip
    env[66] = 0 # Game da ket thuc hay chua
    
    return env
# def visualizeEnv(env_):
#   env = env_.copy()
#   dict_ = {}
#   dict_['Card_on_board'] = env[0:8]
#   for i in range(4):
#       dict_[f'Player_{i}'] = {}
#       dict_[f'Player_{i}']['Cards'] = env[8+i*14:8+i*14+13]
#       dict_[f'Player_{i}']['Chip'] = env[8+i*14+13]
#   return dict_

# ------------------------------------------------------------------------
@njit()
def getAgentSize():
    return AGENT_SIZE

@njit()
def getActionSize():
    return ACTION_SIZE

@njit()
def getStateSize():
    return STATE_SIZE

@njit()
def getAgentState(env):
    state = np.zeros(112)
    pIdx = env[64] # Id player

    # Set player_card
    player_card_idx = env[8+pIdx*14:8+pIdx*14+13]  #idx player card
    state[player_card_idx] = 1

    # Set card on board
    
    card_on_board =  env[0:8]
    card_on_board = card_on_board[card_on_board > -1]
    
   
    card_on_board_id = np.full(52,0)
  
    card_on_board_id[card_on_board]  = 1
    state[52:104] =  card_on_board_id

    state[104] = env[8+pIdx*14+13] #Chip của người chơi
    
    # Lay tong so bai cua 3 nguoi choi khac
    cards_len = list()
    for i in range(getAgentSize()):
        if i == pIdx:
            continue
        cards = env[8+i*14:8+i*14+13]
        len_ = len(np.where(cards > -1)[0])
        cards_len.append(len_)
        
    state[105:108] = cards_len
    state[109] = env[66] # Game da ket thuc hay chua
    chipArr = list()
    for i in range(4):
        if i == pIdx:
            continue
        chip = env[8+i*14+13]
        chipArr.append(chip)
    state[109:112] = chipArr
    return state
# def visualizeState(state_):
#   state = state_.astype(int)

#   p_card_binary = state[0:52]
#   p_card_value = np.where(p_card_binary==1)[0]

#   card_on_board_binary = state[52:104]
#   card_on_board_value = np.where(card_on_board_binary==1)[0]

#   dict_ = {}
#   dict_['Card_on_board'] = {}
#   dict_['Card_on_board']['Value'] = card_on_board_value
#   dict_['Card_on_board']['Binary'] = card_on_board_binary

#   dict_['Player_Cards'] = {}
#   dict_['Player_Cards']['Value'] = p_card_value
#   dict_['Player_Cards']['Binary'] = p_card_binary

#   dict_['Player_chip'] = state[104]
  
#   dict_['Do_Dai_Bai_Cua_Nguoi_Choi_Con_Lai'] = state[105:108] 
#   return dict_

# ------------------------------------------------------------------------


@njit()
def getValidActions(state):

    #Get player card
    p_cards_binary = state[0:52]
    p_cards = np.where(p_cards_binary == 1)[0] 
    # Get card on board

    card_on_board_binary = state[52:104]
    card_on_board = np.where(card_on_board_binary == 1)[0] 
    arr_action = np.full(53,0)
    for i in range(len(arr_action)-1):
      if i in p_cards and i in card_on_board:
        arr_action[i] = 1
      if min(arr_action) == 0:
        arr_action[52] = 1
    return arr_action

# ------------------------------------------------------------------------
@njit()
def stepEnv(action,env):
    player_Id = env[64]
    player_Card = env[8+player_Id*14:8+player_Id*14+13]
    current_card_on_board =  env[0:8]
    if action == 52:
        env[8+player_Id*14+13] -=1 # Tru chip nguoi choi
        if env[8+player_Id*14+13] <= 0:
            return -2
        env[65] += 1 #TempStoreChip += 1
    if action != 53:
        player_Card[np.where(player_Card == action)[0]] = -1
        if action == 7 or action == 20 or action == 33 or action == 46:# action bang 7
            if action == 7:
                current_card_on_board[0:2] = [6,8]
            if action == 20:
                current_card_on_board[2:4] = [19,21]
            if action == 33:
                current_card_on_board[4:6] = [32,34]
            if action == 46:
                current_card_on_board[6:8] = [45,47]
        else: # Check các action hợp lệ
            if 0 <= action < 7:
                current_card_on_board[0] -=1
            if 7 < action < 13:
                current_card_on_board[1] +=1
            if 13 < action < 20:
                current_card_on_board[2] -=1
            if 20 < action < 26:
                current_card_on_board[3] +=1
            if 26 < action < 33:
                current_card_on_board[4] -=1
            if 33 < action < 39:
                current_card_on_board[5] +=1
            if 39 < action < 46:
                current_card_on_board[6] -=1
            if 46 < action < 52:
                current_card_on_board[7] +=1
    env[0:8] = current_card_on_board
    env[8+player_Id*14:8+player_Id*14+13] = player_Card
    return 0

# ------------------------------------------------------------------------

    
@njit()
def stopGame(env):  # Check khi một người chơi hết bài
    for i in range(getAgentSize()):
        p_card = env[8+i*14:8+i*14+13]
        if max(p_card) == -1:
            return i
    return -1
@njit()
def checkEnded(env): # Check khi một người chơi hết chip
    for i in range(getAgentSize()):
        p_chip = env[8+i*14+13] 
        if p_chip <= 0:
            env[66] = 1
            arr_chip = [0,0,0,0]
            for k in range(0,4):
                arr_chip[k] = env[8+k*14+13]
            return np.argmax(arr_chip)
    return False
@njit()
def getReward(state):
    IsEnd = state[109]
    if IsEnd == 0:
        return 0
    if IsEnd == 1:
        p_chip = state[104]
        chip_arr = state[109:112]
        if p_chip >= max(chip_arr):
            return 1
        else:
            return -1
        
    

# ------------------------------------------------------------------------



def one_game(listAgent,perData):

    allGame = True
    agentChipStore = np.array([50,50,50,50])
    idxChip = [21 ,35 ,49,63]
    while allGame:
        env = initEnv()
        env[idxChip] = agentChipStore
        oneGame = True
        if len(listAgent) != 4: # Check list agent
            raise Exception('Phai co 4 nguoi choi')
        while oneGame: 
            count = 10000
            if count > 0:
                for i in range(len(listAgent)):
                    env[64] = i
                    action,perData = listAgent[i](getAgentState(env),perData) # Lấy action của agent
                    player0Chip = stepEnv(action,env) # Xử lý action
                    if player0Chip == -2: # Khi một người chơi hết chip
                        totalChip = env[idxChip]
                        allGame = False
                        winner = np.argmax(totalChip)
                        return winner,perData
                    idxWin1Game = stopGame(env)

                if idxWin1Game != -1: # Khi một người chơi hết bài và 1 trận kết thúc
                    agentChipStore[idxWin1Game] += env[65]# temStoreChip
                    agentChipStore = env[idxChip] 
                    oneGame = False
                    # 
                count -= 1
            if count < 0:
                return 5,perData
@njit()
def numba_one_game(p0,p1,p2,p3,perData,pIOrder):
    allGame = True
    agentChipStore = np.array([50,50,50,50])
    idxChip = [21 ,35 ,49,63]
    while allGame:
        env = initEnv()
        env[idxChip] = agentChipStore
        oneGame = True
        while oneGame: 
            count = 10000
            if count > 0:
                for i in range(getAgentSize()):
                    env[64] = i
                    state = getAgentState(env)
                    if pIOrder[i] == 0:
                        action,perData = p0(state)
                    if pIOrder[i] == 1:
                        action,perData = p1(state)
                    if pIOrder[i] == 2:
                        action,perData = p2(state)
                    if pIOrder[i] == 3:
                        action,perData = p3(state)
                        
                    player0Chip = stepEnv(env,action) # Xử lý action
                    if player0Chip == -2: # Khi một người chơi hết chip
                        totalChip = env[idxChip]
                        allGame = False
                        winner = np.argmax(totalChip)
                        return winner,perData
                    idxWin1Game = stopGame(env)

                if idxWin1Game != -1: # Khi một người chơi hết bài và 1 trận kết thúc
                    agentChipStore[idxWin1Game] += env[65]# temStoreChip
                    agentChipStore = env[idxChip] 
                    oneGame = False
                    # 
                count -= 1
            if count < 0:
                return 5,perData
# ------------------------------------------------------------------------

def normal_main(listAgent,num_match,perData):
    numWin = [0,0,0,0,0]
    pIdOrder = np.arange(4)
    for _ in range(num_match):
      np.random.shuffle(pIdOrder)
      shuffledListAgent = [listAgent[i] for i in pIdOrder]
      winner,perData = one_game(shuffledListAgent,perData)
      numWin[pIdOrder[winner]] += 1
    return numWin,perData

@jit
def numba_main(p0, p1, p2, p3, times, perData):
    numWin = np.full(5 , 0)
    pIdOrder = np.arange(4)
    for _ in range(times):
      np.random.shuffle(pIdOrder)
      winner,perData = numba_one_game(p0,p1,p2,p3,perData,pIdOrder)
      numWin[pIdOrder[winner]] += 1 
    return numWin,perData

def randomBot(state,perData):
    

    actions_binary = getValidActions(state)
    idx_valid_action = np.where(actions_binary==1)[0]    
    #Get and handle card on board
    idx = np.random.randint(0,len(idx_valid_action))
    return idx_valid_action[idx], perData
@njit()
def numbaRandomBot(state,perData):
    actions_binary = getValidActions(state)
    idx_valid_action = np.where(actions_binary==1)[0]    
    #Get and handle card on board
    idx = np.random.randint(0,len(idx_valid_action))
    return idx_valid_action[idx], perData
# listAgent = [numbaRandomBot,numbaRandomBot,numbaRandomBot,numbaRandomBot]
# win,perData = normal_main(listAgent,5,perData)
# print('win',win)
@jit
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    allGame = 1
    agentChipStore = np.array([50,50,50,50])
    idxChip = [21 ,35 ,49,63]
    while allGame:
        env = initEnv()
        env[idxChip] = agentChipStore
        oneGame = 1
        while oneGame: 
            count = 10000
            if count > 0:
                for i in range(len(list_other)):
                    env[64] = i
                    idx = env[64]
                    player_state = getAgentState(env)
                    if list_other[idx] == -1:
                        action, per_player = p0(player_state,per_player)
                    elif list_other[idx] == 1:
                        action, per1 = p1(player_state,per1)
                    elif list_other[idx] == 2:
                        action, per2 = p2(player_state,per2)
                    elif list_other[idx] == 3:
                        action, per3 = p3(player_state,per3)    
                    stepEnv(action, env)
                    player0Chip = stepEnv(env,action) # Xử lý action 
                    if player0Chip == -2: # Khi một người chơi hết chip
                        # totalChip = env[idxChip]
                        allGame = 0
                        winner = checkEnded(env)
                        if  np.where(list_other == -1)[0] == winner:
                             return True,per_player
                    idxWin1Game = stopGame(env)

                if idxWin1Game != -1: # Khi một người chơi hết bài và 1 trận kết thúc
                    agentChipStore[idxWin1Game] += env[65]# temStoreChip   
                    agentChipStore = env[idxChip] 
                    oneGame = 0
                    # 
                count -= 1
            if count < 0:
                return False,per_player
@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per
@jit()
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, 
p3):
 win = 0
 for _n in range(num_game):
    np.random.shuffle(list_other)
    winner,per_player = one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3)
    win += winner
 return win, per_player

import importlib.util, json, sys
from setup import SHOT_PATH

def load_module_player(player):
 return importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()

@jit
def numba_main_2(p0, n_game, per_player, level, *args):
  list_other = np.array([1, 2, 3, -1])
  if level == 0:
    per_agent_env = np.array([0])
    return n_game_numba(p0, n_game, per_player, list_other, per_agent_env, per_agent_env, per_agent_env, random_Env, random_Env, random_Env)
  else:
    env_name = sys.argv[1]
    if len(args) > 0:
      dict_level = json.load(open(f'{SHOT_PATH}Log/check_system_about_level.json'))
    else:
      dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))

    if str(level) not in dict_level[env_name]:
      raise Exception('Hiện tại không có level này') 

    lst_agent_level = dict_level[env_name][str(level)][2]
    p1 = load_module_player(lst_agent_level[0]).Test
    p2 = load_module_player(lst_agent_level[1]).Test
    p3 = load_module_player(lst_agent_level[2]).Test
    per_level = []
    for id in range(getAgentSize()-1):
      data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{lst_agent_level[id]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True))
      per_level.append(data_agent_env)
    return n_game_numba(p0, n_game, per_player, list_other, per_level[0], per_level[1], per_level[2], p1, p2, p3)


