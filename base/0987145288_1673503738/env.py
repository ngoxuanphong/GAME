import numpy as np
from numba import njit, jit
from numba.typed import List

@njit
def getActionSize():
  return 18
@njit
def getAgentSize():
  return 4
@njit
def getStateSize():
  return 32

@njit
def initEnv():
  all = np.arange(52) % 13
  np.random.shuffle(all[:-1]) 

  env = np.full(118, 0) # Khởi tạo env
  env[:52] = all        #trộn bộ bài
  env[52] = 52 #số lá ở trên bàn
  for k in range(4):    # thông tin của người chơi k
    for i in env[k*5: k*5 + 5]:
      idx = 53+ k*15
      env[idx + i] += 1
      env[52] -= 1
    score = len( np.where(env[idx: idx +13] == 4)[0] )
    env[67+ k*15] = score
    env[66+ k*15] = sum(env[idx: idx+13]) - 4*score

  env[116] = -1 #chưa yêu cầu lá
  
  '''
  env[113] = turn  
  env[114] = phase
  env[115] = người bị yêu cầu ( =0 nếu chưa yêu cầu ai)
  env[117] = game end chưa
  '''
  return env

def visualizeEnv(env):
  dict_ = {}
  dict_["Những lá còn lại"] = env[52-env[52]: 52]
  # for i in range(4):
  #   dict_[f'Player_{i}'] = env[53+i*15: 68+i*15]
  dict_["players_"] = env[53:113].reshape(4,15)
  dict_["Số lá còn lại"] = env[52]
  dict_["Turn"] = env[113:118]
  return dict_

@njit
def getAgentState(env):
  state = np.zeros(32)

  pIdx = env[113] % 4 # Index của người chơi nhận state
  for i in range(4): # Sắp xếp lại thông tin env theo góc nhìn người chơi
    pEnvIdx = (pIdx + i) % 4
    if i == 0:
      state[0:15] = env[53 + pEnvIdx*15: 68 + pEnvIdx*15]
    else: 
      arr = env[53 + pEnvIdx*15: 68 + pEnvIdx*15]
      state[12 + i*3] = arr[13] #số lá
      state[13 + i*3] = arr[14] #điểm
      state[14 + i*3] = np.where(arr[:13]==4)[0][-1]
      
  if env[52]:  #lá bài còn lại trên bài không
    state[24] = 1

  state[25 + env[114]] = 1 #phase
  state[28 + env[115]] = 1 #người bị yêu cầu
  state[31] = env[117]
  return state

def visualizeState(state):
    dict_ = {}
    dict_["InforPlayer"] = state[0:15]
    for i in range(1,4):
        dict_[f'Player_{i}'] = state[12+i*3: 15+i*3]
    dict_["onBoard"] = state[24]
    dict_["Phase"] = state[25:28]
    dict_["Người bị yêu cầu"] = state[28:31]
    dict_["Game end chưa"] = state[31]
    return dict_

@njit
def getValidActions(state):
  validActions = np.full(18,0)
  phase = np.where(state[25:28]>0)[0]
  if phase == 0:
    for i in range(3):
      if state[15 + 3*i] > 0: #người bị yêu cầu phải còn bài trên tay
        validActions[i+1]=1
  elif phase == 1:
    arr = state[0:13]
    idx = np.where( (arr>0)&(arr<4) )[0]
    validActions[idx+4] = 1
  elif phase == 2:
    validActions[0] = 1

  if len(np.where(validActions == 1)[0]) == 0:#lấy perData lúc endgame
    validActions[17] = 1 
  return validActions

@njit
def stepEnv(action, env):
  if action == 0: #boc
    player_0 = env[113] % 4
    arr_0 = env[53 + 15 * player_0: 68 + 15 * player_0] #arrcủa pl_0

    l = env[52 - env[52]] #lá vừa bốc
    env[52] -= 1
    arr_0[l] += 1 #đưa lá đó cho người chơi
    #tinh lai diem
    arr_0[13] += 1
    if arr_0[l] == 4:
      arr_0[14] += 1 #điểm tăng 1
      arr_0[13] -= 4
    
    #neu het bai tren tay thi phai lay du 5 la tu bo bai
    if env[66 + player_0*15]==0:
      while env[52]>0 and env[66 + player_0*15] <= 5:
        l_ = env[52 - env[52]]
        env[52] -= 1
        arr_0[l_] += 1
        arr_0[13] += 1
        #tinh diem
        if arr_0[l_] == 4:
          arr_0[14] += 1 #điểm tăng 1
          arr_0[13] -= 4

    if l == env[116] and arr_0[13] != 0: #bốc lá cuối cùng giống cây yêu cầu và hết bài luôn
      env[114:117] = np.array([0,0,-1])
    else:
      if np.sum(env[np.array([66,81,96,111])]) - arr_0[13] != 0: #nhung nguoi choi khac con bai
        env[114:117] = np.array([3,0,-1])
      elif env[52]!= 0:
        env[114:117] = np.array([2,0,-1])

  elif action < 4:
    env[114] = 1 #phase = 1 yêu cầu lá bài
    env[115] = action #người yêu cầu
  elif action < 17:
    idx = action - 4 #index là bài
    player_0 = env[113] % 4 #người chơi trong turn đó
    player_1 = (player_0 + env[115]) % 4 #người bị yêu cầu

    arr_0 = env[53 + 15 * player_0: 68 + 15 * player_0] #arr lá bài của pl_0
    arr_1 = env[53 + 15 * player_1: 68 + 15 * player_1] #arr lá bài của pl_1
    
    if idx in np.where((arr_1[:13] > 0) & (arr_1[:13] < 4))[0]:
      arr_0[idx] += arr_1[idx] #đưa thẻ
      arr_1[13] -= arr_1[idx]
      arr_0[13] += arr_1[idx]
      arr_1[idx] = 0
      #tinh lai diem
      if arr_0[idx] == 4:
          arr_0[14] += 1 #điểm tăng 1
          arr_0[13] -= 4

      if env[66 + player_0*15] == 0:
        while env[52]>0 and env[66 + player_0*15] <= 5:
          l_ = env[52 - env[52]]
          env[52]-=1
          arr_0[l_] += 1
          arr_0[13] += 1
          #tinh diem
          if arr_0[l_] == 4:
            arr_0[14] += 1 #điểm tăng 1
            arr_0[13] -= 4

      if arr_0[13] != 0:
        if np.sum(env[np.array([66,81,96,111])]) - arr_0[13] != 0:
          env[114:117] = np.array([0,0,-1])
        elif env[52]!= 0:
          env[114:117] = np.array([2,0,-1])
      else:
        env[114:117] = np.array([3,0,-1])
      
    else:
        env[114] = 0 #xóa người bị yêu cầu
        if env[52] :
            env[114] = 2
            env[116] = idx #nhớ lá vừa yêu cầu
        else:
            env[114] = 3

#phase = 3------------------
  if (env[67] + env[82] + env[97] + env[112]) != 13:
    while env[114] ==3:
      env[114:117] = np.array([0,0,-1]) #xóa thông tin turn trước
      env[113] = env[113] + 1 #chuyển sang người chơi khác
      new_pl = env[113] % 4
      if env[66 + new_pl*15] == 0 : #người chơi này hết bài 
        env[114] = 3
  else:
    env[117]=1

@njit
def checkEnded(env):
  scoreArr = env[np.array([67, 82 , 97 , 112])]
  max_ = np.max(scoreArr)

  if sum(scoreArr) == 13:
    env[117] = 1
    max_point = np.zeros(4)
    for _ in range(4):
      arr = env[53 + _*15: 66 + _*15]
      max_point[_] = np.where(arr == 4)[0][-1] 
    
    arr_win = np.where(scoreArr == max_)[0]
    m = np.max(max_point[arr_win])
    for k in arr_win:
      if max_point[k] == m:
        return k
  else:
    return -1
    
@njit
def getReward(state):
  if state[31]==0:
    return -1
  else:
    if state[14] > np.max(state[16:24:3]):
      return 1
    elif state[14] == np.max(state[16:24:3]):
      arr_win = np.where(state[16:24:3] == state[14])[0]
      point_pl = np.where(state[:13]==4)[0][-1]
      if point_pl > np.max(state[17 + 3*arr_win]):
        return 1
      else:
        return 0
    else:
      return 0
    
def randomBot(state, perData):
  validActions = getValidActions(state)
  validActions = np.where(validActions==1)[0]
  idx = np.random.randint(0, len(validActions))
  return validActions[idx], perData
  
def one_game(listAgent, perData):#----------------------------------------------
  env = initEnv()
  # print(visualizeEnv(env))
  winner = -1
  # turn = -1
  while env[113] < 400:
    pIdx = env[113] % 4
    # if turn != pIdx :
    #   turn = pIdx
    #   print("--------Turn =", env[113])
    action, perData = listAgent[pIdx](getAgentState(env), perData)
    # print(action)
    stepEnv(action, env)
    # print(visualizeEnv(env))
    winner = checkEnded(env)
    if winner != -1:
        break
  for pIdx in range(4):
      env[113] = pIdx
      action, perData = listAgent[pIdx](getAgentState(env), perData)
  return winner, perData

def normal_main(listAgent, times, perData):#------------------------------------
  # if len(listAgent) != 4:
  #       raise Exception('Hệ thống chỉ cho phép có đúng 4 người chơi!!!')
    
  numWin = [0, 0, 0, 0, 0]
  pIdOrder = np.arange(4)
  for _ in range(times):
    # if printMode and _ != 0 and _ % k == 0:
    #     print(_, numWin)

    np.random.shuffle(pIdOrder)
    # print(pIdOrder)
    shuffledListAgent = [listAgent[i] for i in pIdOrder]
    winner, perData = one_game(shuffledListAgent, perData)

    if winner == -1:
        numWin[4] += 1
    else:
        numWin[pIdOrder[winner]] += 1
  
  # if printMode:
  #   print(_+1, numWin)

  return numWin, perData

@njit
def numbaRandomBot(state, perData):
  validActions = getValidActions(state)
  validActions = np.where(validActions==1)[0]
  idx = np.random.randint(0, len(validActions))
  return validActions[idx], perData

@njit
def numba_one_game(p0, p1, p2, p3, perData, pIdOrder):
  env = initEnv()

  winner = -1
  while env[113] < 400:
    pIdx = env[113] % 4
    if pIdOrder[pIdx] == 0:
      action, perData = p0(getAgentState(env), perData)
    elif pIdOrder[pIdx] == 1:
      action, perData = p1(getAgentState(env), perData)
    elif pIdOrder[pIdx] == 2:
      action, perData = p2(getAgentState(env), perData)
    elif pIdOrder[pIdx] == 3:
      action, perData = p3(getAgentState(env), perData)
    
    stepEnv(action, env)
    winner = checkEnded(env)
    if winner != -1:
        break
  
  for pIdx in range(4):
    env[113] = pIdx
    if pIdOrder[pIdx] == 0:
      action, perData = p0(getAgentState(env), perData)
    elif pIdOrder[pIdx] == 1:
      action, perData = p1(getAgentState(env), perData)
    elif pIdOrder[pIdx] == 2:
      action, perData = p2(getAgentState(env), perData)
    elif pIdOrder[pIdx] == 3:
      action, perData = p3(getAgentState(env), perData)

  return winner, perData

@njit
def numba_main(p0, p1, p2, p3, times, perData):
  numWin = np.full(5, 0)
  pIdOrder = np.arange(4)
  for _ in range(times):
    # if printMode and _ != 0 and _ % k == 0:
    #   print(_, numWin)

    np.random.shuffle(pIdOrder)
    winner, perData = numba_one_game(p0, p1, p2, p3, perData, pIdOrder)

    if winner == -1:
      numWin[4] += 1
    else:
      numWin[pIdOrder[int(winner)]] += 1
  
  # if printMode:
  #     print(_+1, numWin)

  return numWin, perData

@jit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
  env = initEnv()
  while env[113] < 400:
    idx = env[113]%4
    player_state = getAgentState(env)
    list_action = getValidActions(player_state)

    if list_other[idx] == -1:
      action, per_player = p0(player_state,per_player)
    elif list_other[idx] == 1:
      action, per1 = p1(player_state,per1)
    elif list_other[idx] == 2:
      action, per2 = p2(player_state,per2)
    elif list_other[idx] == 3:
      action, per3 = p3(player_state,per3)

    if list_action[action] != 1:
      raise Exception('Action không hợp lệ')
    stepEnv(action, env)
    if checkEnded(env) != -1:
      break

  turn = env[113]
  for idx in range(4):
    env[113] = idx
    if list_other[idx] == -1:
      p_state = getAgentState(env)
      p_state[31] = 1
      act, per_player = p0(p_state, per_player)

  env[113] = turn
  winner = 0
  if np.where(list_other == -1)[0] == checkEnded(env): winner = 1
  else: winner = 0
  return winner, per_player

@njit()
def random_Env(p_state, per):
  arr_action = getValidActions(p_state)
  arr_action = np.where(arr_action == 1)[0]
  act_idx = np.random.randint(0, len(arr_action))
  return arr_action[act_idx], per

@jit()
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
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

# listAgent = [randomBot,randomBot,randomBot,randomBot]
# perData = []
# perData.append(np.array([[0.]]))

# # normal_main(listAgent, 100, perData)
# print(numba_main(numbaRandomBot, numbaRandomBot, numbaRandomBot, numbaRandomBot, 100, perData))
