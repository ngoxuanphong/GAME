import numpy as np
import random as rd
from numba import njit, jit
import sys, os
from setup import SHOT_PATH
import importlib.util
game_name = sys.argv[1]

def setup_game(game_name):
    spec = importlib.util.spec_from_file_location('env', f"{SHOT_PATH}base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module

env = setup_game(game_name)

getActionSize = env.getActionSize
getStateSize = env.getStateSize
getAgentSize = env.getAgentSize

getValidActions = env.getValidActions
getReward = env.getReward

normal_main = env.normal_main
numba_main_2 = env.numba_main_2

from numba.typed import List

@njit
def DataAgent():
    return [List(np.array([np.zeros((getStateSize(),getActionSize()))])),List(np.array([np.zeros((getStateSize(),getActionSize()))])), List(np.array([np.zeros((getStateSize(),getActionSize()))])),List(np.array([np.zeros((getStateSize(),getActionSize()))]))]

@njit 
def Train(state,per):
  weight = np.random.choice(getActionSize(), size=getActionSize(), replace=False)
  actions = getValidActions(state)  
  output = weight*actions+actions
  action = np.argmax(output)
  # TÌm các max value cho array bias 
  if np.max(per[3][0])> 5000 :    
    for i in range(len(state)):
      if int(state[i])> np.max(per[2][0]):
        k = int(state[i]-np.max(per[2][0]))
        per[2][0] += k
        for i in range(k):
          per[1].append(np.zeros((getStateSize(),getActionSize())))
          per[0].append(np.zeros((getStateSize(),getActionSize())))
          per[2].append(np.zeros((getStateSize(),getActionSize())))
          per[3].append(np.zeros((getStateSize(),getActionSize())))
      if state[i]<0:
        pass
      per[1][int(state[i])][int(i)] += weight
  elif np.max(per[3][0]) == 5000 :
    per[3][0] += 2000 
    for i in range(int(np.max(per[2][0]))):
      per[1].append(np.zeros((getStateSize(),getActionSize())))
      per[0].append(np.zeros((getStateSize(),getActionSize())))
      per[2].append(np.zeros((getStateSize(),getActionSize())))
      per[3].append(np.zeros((getStateSize(),getActionSize())))
  else:
      if np.max(state) > np.max(per[2][0]) :
          o = np.max(state) - np.max(per[2][0]) 
          per[2][0] += int(o)

  # Bắt đầu lưu array bias
  if np.max(per[3][0])>5000:    
    if getReward(state) == 1:
      for i in range(int(np.max(per[2][0]))):
        per[0][i] += per[1][i]
      for i in range(int(np.max(per[2][0]))):
        per[1][i] = np.zeros((getStateSize(),getActionSize()))
    if getReward(state) == 0 :
      for i in range(int(np.max(per[2][0]))):
        per[1][i] = np.zeros((getStateSize(),getActionSize()))
  else:
    if getReward(state) != -1:
      per[3][0] += 1

  return action,per

@njit 
def Test(state,per):
  weight =np.zeros((getActionSize()))
  for i in range(len(state)):
    if int(state[i])> np.max(per[2][0]) :
      pass
    elif np.sum(per[0][int(state[i])][int(i)]) == 0:
      pass
    elif  state[i]<0:
      pass
    else:
      weight += (per[0][int(state[i])][int(i)]/np.max(per[0][int(state[i])][int(i)]))

  actions = getValidActions(state) 
  output = weight*actions+actions
  action = np.argmax(output)
  return action,per