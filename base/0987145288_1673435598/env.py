import numpy as np
from numba import njit,jit
import numba
from numba.typed import List
import sys
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
@njit
def initEnv():
    env = np.zeros(80)
    card = np.arange(52)#card
    np.random.shuffle(card)
    for i in range(4):
        env[card[i*8:(i+1)*8]] = i+1
    env[52] = card[-1] #trump suit card
    env[53] = 0 #mode: attack or defense
    env[54:57] = [1,3,4]
    env[57]= 1 #num of people choose attack this round
    env[58] = 2 #player_id defending
    env[59] = 0 #index player attack in env[54:57]
    env[60:80] = card[32:52] #card on deck
    return env
@jit
def getStateSize():
    return 166
@jit
def getAgentSize():
    return 4
@jit
def getAgentState(env):
    state = np.zeros(getStateSize())
    if env[53]==0:
        player_id = env[54:57][int(env[59])] #attack player id
    elif env[53]==1:
        player_id = env[58] #defend player id
    state[0:52][np.where(env[0:52]==player_id)] = 1 # card player hold
    state[52:104][np.where(env[0:52]==6)] = 1 #all card defender defend this round
    state[104:156][np.where(env[0:52]==5)] = 1#card have to defend this round
    if env[53]==1:
        state[156:158] = [0,1] # attack, defend
    elif env[53]==0:
        state[156:158] = [1,0]
    state[158:162][int(env[52])//13] = 1 #trump suit
    state[162] = len(np.where(env[0:52]==0)[0]) #num card on deck
    k = 0
    for i in range(1,5):
        if i==player_id:
            pass
        else:
            state[163+k] = np.where(env[0:52]==i)[0].shape[0]  #num card other player have on hand
            k+=1
    return state

@jit
def getActionSize():
    return 53
@jit
def getDefenseCard(state):
    card = np.zeros(52)
    idx = np.argmax(state[158:162]) #trump suit: 0:spade,1:club,2:diamond,3:heart
    card_def_id = np.argmax(state[104:156])
    if card_def_id//13!=idx:#card have to defend not a trump card
        card[13*idx:13*(idx+1)][np.where(state[13*idx:13*(idx+1)]==1)] = 1 # trump card on hand
        card[card_def_id+1:13*(card_def_id//13+1)][np.where(state[card_def_id+1:13*(card_def_id//13+1)]==1)] = 1 #same type card, higher value on hand.
    else:#card have to defend is a trump card
        card[card_def_id+1:13*(idx+1)][np.where(state[card_def_id+1:13*(idx+1)]==1)] = 1 #higher value trump card only.
    return card
    
@jit
def getAttackCard(state):
    card = np.zeros(52)
    card_on_board = np.where(state[52:104]==1)[0]
    card_value_on_board = card_on_board % 13 #value of that card (ex: 4 diamond is 4)
    card_on_hand = np.where(state[0:52]==1)[0] #card on player's hand
    for c in card_on_hand:
        if c%13 in card_value_on_board:
            card[c] = 1
    return card
@jit
def getValidActions(state):
    list_action = np.zeros(getActionSize())
    #attack
    if state[156]==1 and np.sum(state[52:104])==0: #main attacker, defender have nothing to defend yet.
        list_action[0:52] = state[0:52]
    elif state[156]==1 and np.sum(state[52:104])!=0:#side attacker, attack only card with same value on the defend board( 4 heart on hand if have 4 spade on board)
        list_action[0:52] = getAttackCard(state)
        list_action[52] = 1
    #defense
    if state[157]==1:#defender
        list_action[0:52] = getDefenseCard(state)
        list_action[52] = 1
    if np.sum(list_action)==0:
        list_action[52] = 1
    return list_action
@jit
def drawCard(env):
    turn_draw_card = np.zeros(4)
    turn_draw_card[np.array([0,2,3])] = env[54:57] #attack player,main attack draw first.
    turn_draw_card[1] = env[58] #defend player draw second.
    for p_id in turn_draw_card: #draw card
        num_card_on_deck = len(np.where(env[0:52]==0)[0])#num cards left on deck
        if num_card_on_deck > 0:
            num_card_player = len(np.where(env[0:52]==p_id)[0])
            if num_card_player < 8:
                num_card_need = 8 - num_card_player
                if num_card_on_deck >= num_card_need:
                    env[env[60:80].astype(np.int64)[20-num_card_on_deck:20-num_card_on_deck+num_card_need]] = p_id
                else:
                    env[env[60:80].astype(np.int64)[20-num_card_on_deck:]] = p_id
    return env
@jit
def changeAttackPlayer(env): #change the defender and attacker
    if env[58]==1:
        env[54:57] = [4,2,3]
    elif env[58]==2:
        env[54:57] = [1,3,4]
    elif env[58]==3:
        env[54:57] = [2,4,1]
    elif env[58]==4:
        env[54:57] = [3,1,2]
@jit
def stepEnv(action,env):
    if action == 52:#skip
        if env[53] == 1: #defense
            env[0:52][np.where(env[0:52]==5)] = env[58] #Attacker hold all card
            env[0:52][np.where(env[0:52]==6)] = env[58] #Attacker hold all card
            env = drawCard(env) #draw card
            env[58] = (env[58]+2)%4 if env[58] > 2 else env[58]+2 #change defend player
            changeAttackPlayer(env) #change attack player
            env[53] = 0 #reset mode: attack
            env[59] = 0
        elif env[53] == 0:#attack
            env[57] += 1 #num attacker skip this round
            env[59]  = (env[59]+1)%3
            if env[57] == 3:#all attacker skip this round
                env[0:52][np.where(env[0:52]==5)] = -1#Thrown away card
                env[0:52][np.where(env[0:52]==6)] = -1#Thrown away card
                env = drawCard(env) #draw card
                env[58] = 1 if env[58]==4 else env[58]+1 #change defend player
                changeAttackPlayer(env) #change attack players
                env[57] = 0
                env[59] = 0
                
    else:#attack or defend any card
        if env[53] == 1:#defense
            env[0:52][np.where(env[0:52]==5)] = 6 #defense this card successful
            env[action] = 6
            env[53] = 0#change mode: attack
        elif env[53] == 0: #attack
            env[action] = 5 #this card have to defend
            env[53] = 1 #change mode: defend
            env[57] = 0 #change num player attack skip turn to 0
            env[59]  = (env[59]+1)%3 #change attack player
            
    # return env


@jit
def checkEnded(env):
    if len(np.where(env[0:52]==0)[0])==0:#if no card left on deck
        list_win = []
        turn_draw_card = np.zeros(4)
        turn_draw_card[np.array([0,2,3])] = env[54:57]
        turn_draw_card[1] = env[58]
        for p_id in turn_draw_card:
            if len(np.where(env[0:52]==p_id)[0]) == 0: #if player have no card left
                list_win.append(p_id)
            else:
                pass
        if len(list_win)>0:
            return int(list_win[0]-1)
        else:
            return -1
    return -1

@jit
def getReward(state):
    if state[162]==0:
        if np.sum(state[0:52])==0:
            return 1
        elif np.min(state[163:166])==0:
            return 0
        else:
            return -1
    else:
        return -1


    
def one_game(listAgent,perData):
    env = initEnv()
    winner = -1
    turn = 0
    while True:
        turn +=1
        if env[53]==1:#defense
            pIdx = int(env[58] - 1)
        else:#attack
            pIdx = int(env[54:57][int(env[59])] - 1)
        action, perData = listAgent[pIdx](getAgentState(env), perData)
        stepEnv(action, env)
        # #print(env[:53])
        winner = checkEnded(env)
        if winner != -1:
            break
    for i in range(4):
        env[53] = 1
        env[58] = i
        action, perData = listAgent[i](getAgentState(env), perData)
    return winner, perData
@jit()
def numba_one_game(p0, p1, p2, p3, perData, pIdOrder):
    env = initEnv()
    winner = -1
    while True:
        if env[53]==0:
            pIdx = int(env[58] - 1)
        else:
            pIdx = int(env[54:57][int(env[59])] - 1)
        try:
            if pIdOrder[pIdx] == 0:
                action, perData = p0(getAgentState(env), perData)
            elif pIdOrder[pIdx] == 1:
                action, perData = p1(getAgentState(env), perData)
            elif pIdOrder[pIdx] == 2:
                action, perData = p2(getAgentState(env), perData)
            elif pIdOrder[pIdx] == 3:
                action, perData = p3(getAgentState(env), perData)
        except:
            #print(list(env))
            break
        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break
    
    for p_idx in range(4):
        env[53] = 1
        env[58] = p_idx
        p_state = getAgentState(env)
        if pIdOrder[p_idx] == 0:
            act, perData = p0(p_state, perData)
        elif pIdOrder[p_idx] == 1:
            act, perData = p1(p_state, perData)
        elif pIdOrder[p_idx] == 2:
            act, perData = p2(p_state, perData)
        elif pIdOrder[p_idx] == 3:
            act, perData = p3(p_state, perData)
    return winner, perData
        
# @jit
def normal_main(listAgent, times, perData):
    numWin = [0,0,0,0,0]
    pIdOrder = [0,1,2,3]
    for _ in range(times):
        np.random.shuffle(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = one_game(shuffledListAgent, perData)
        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    return numWin, perData
@jit()
def numba_main(p0, p1, p2, p3, times, perData):
    numWin = [0,0,0,0,0]
    pIdOrder = [0,1,2,3]
    for _ in range(times):
        np.random.shuffle(pIdOrder)
        winner, perData = numba_one_game(p0, p1, p2, p3, perData, pIdOrder)
        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    return numWin, perData


@jit()
def one_game_numba(p0,pIdOrder,per_player,per1,per2,per3,p1,p2,p3):
    env = initEnv()
    winner = -1
    turn = 0
    while True:
        turn +=1
        if env[53]==0:
            pIdx = int(env[58] - 1)
        else:
            pIdx = int(env[54:57][int(env[59])] - 1)
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env), per3)
        stepEnv(action,env)
        
        winner = checkEnded(env)
        if winner != -1:
            break
    for idx in range(4):
        if pIdOrder[idx] == -1:
            env[53] = 1
            env[58] = idx * 1.0
            p_state = getAgentState(env)
            act, per_player = p0(p_state, per_player)

    win = False        
    if np.where(pIdOrder == -1)[0][0] == checkEnded(env): 
        win = True
    else: 
        win = False

    
            # print('ok')
    return win, per_player

@jit()
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3)
        # print(winner)
        win += winner
    return win, per_player

import importlib.util, json, sys
from setup import SHOT_PATH

def load_module_player(player):
    return  importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()

@jit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per

# @jit()
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
        
# @jit
def random_player1(state,per):
    list_action  = np.where(getValidActions(state)==1)[0]
    action = np.random.choice(list_action)
    if getReward(state) != -1:
        per[0] += 1

    return action,per
# @jit
def random_player(state,per):
    list_action  = np.where(getValidActions(state)==1)[0]
    action = np.random.choice(list_action)
    return action,per