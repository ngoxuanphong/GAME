
# from system.mainFunc import dict_game_for_player, load_data_per2
# game_name_ = 'Catan_v2'
# import random
# import numpy as np
# from setup import game_name,time_run_game
# from numba import jit, njit, prange
# import warnings
# from numba.typed import List
# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
# warnings.simplefilter('ignore', category=NumbaWarning)

# ##########################################################

# @njit()
# def data_to_layer_NhatAnh_130922(state,data0, data1):
#     state = np.dot(state,data0)
#     state *= state > 0
#     active = state > 0
#     layer1 = data1.flatten() * active
#     return layer1

# @njit()
# def test2_NhatAnh_130922(state,file_per_2):
#     layer = np.zeros(getActionSize())
#     for id in range(len(file_per_2[0])):
#         layer += data_to_layer_NhatAnh_130922(state,file_per_2[0][id], file_per_2[1][id])
#     base = np.zeros(getActionSize())
#     actions = getValidActions(state)
#     actions = np.where(actions == 1)[0]
#     for act in actions:
#         base[act] = 1
#     layer *= base
#     base += layer
#     action = np.random.choice(np.where(base == np.max(base))[0])
#     return action

# ###########################################################

# @njit()
# def basic_act_NhatAnh_200922(state,base):
#     actions = getValidActions(state)
#     actions = np.where(actions == 1)[0]
#     for act in base:
#         if act in actions:
#             return act
#     ind = np.random.randint(len(actions))
#     action = actions[ind]
#     return action

# @njit()
# def test2_NhatAnh_200922(state,file_per_2):
#     action = basic_act_NhatAnh_200922(state,file_per_2)
#     return action

# ###########################################################

# @njit()
# def advance_act_NhatAnh_270922(state,data):
#     for id in range(len(data[1])):
#         x = data[1][id].reshape(len(data[1][id]), 1)
#         mt = np.dot(state,x)
#         if mt[0] <= 0:
#             action = basic_act_NhatAnh_200922(state,data[0][id-1])
#             return int(action)
#         else:
#             action = basic_act_NhatAnh_200922(state,data[0][id])
#             return int(action)
#     return np.random.choice(np.where(getValidActions(state) == 1)[0])

# @njit()
# def test2_NhatAnh_270922(state, file_per_2):
#     action = advance_act_NhatAnh_270922(state,file_per_2)
#     return action

# #################################################################
# #################################################################
# #################################################################
# @njit()
# def _sigmoid_khanh_130922_(x):
#     sig = 1 / (1 + np.exp(-x))
#     return sig

# @njit()
# def _silu_khanh_130922_(x, theda = 1.0):
#     return x * _sigmoid_khanh_130922_(theda *x)

# @njit()
# def neural_network_khanh_130922(play_state, file_temp0, file_temp1, file_temp2):
#     if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
#         matran1 = np.dot(play_state,file_temp0)
#         matran1 = 1 / (1 + np.exp(-matran1))
#         matran21 = np.dot(matran1,file_temp1)
#         matran21 *= (matran21 > 0)
#         matran2 = np.dot(matran21, file_temp2)
#         return matran2    
#     elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
#         matrix1 = np.dot(play_state, file_temp0)
#         matrixRL1 = 1 / (1 + np.exp(-matrix1))
#         matrix2 = np.dot(matrixRL1, file_temp1)
#         matrixRL2 = 1 / (1 + np.exp(-matrix2))
#         all_action_val = np.dot(matrixRL2, file_temp2 )
#         return all_action_val  
#     elif 170 < len(play_state) < 250  : #SHERIFF 
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = _silu_khanh_130922_(matran1, theda = 1.0)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
#     else :#SUSHIGO-main, MACHIKOR0
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 *= (matran1 > 0)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2

# @njit()
# def neural_network_khanh_130922_2(play_state, file_temp0, file_temp1):
#     if 170 < len(play_state) < 250  : #SHERIFF 
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = _silu_khanh_130922_(matran1, theda = 1.0)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
#     else :#SUSHIGO-main, MACHIKOR0
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 *= (matran1 > 0)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2

# @njit()
# def test2_Khanh_130922(play_state, file_per_2):
#     a = getValidActions(play_state)
#     a = np.where(a == 1)[0]
#     if len(file_per_2) == 3:
#         matran2 = neural_network_khanh_130922(play_state, file_per_2[0], file_per_2[1], file_per_2[2])
#     else:
#         matran2 = neural_network_khanh_130922_2(play_state, file_per_2[0], file_per_2[1])
#     max_ = 0
#     action_max = a[random.randrange(len(a))]
    
#     for act in a:
#         if matran2[act] > max_:
#             max_ = matran2[act]
#             action_max = act
#     return action_max

# ################################################################

# @njit()
# def relu6_khanh_200922(x):
#     return np.minimum(np.maximum(0, x),6)

# @njit()
# def neural_network_khanh_200922(play_state, file_temp0, file_temp1, file_temp2):
#     if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
#         matran1 = np.dot(play_state,file_temp0)
#         matran1 = 1 / (1 + np.exp(-matran1))
#         matran21 = np.dot(matran1,file_temp1)
#         matran21 *= (matran21 > 0)
#         matran2 = np.dot(matran21, file_temp2)
#         return matran2    
#     elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
#         matrix1 = np.dot(play_state,file_temp0)
#         matrixRL1 = 1 / (1 + np.exp(-matrix1))
#         matrix2 = np.dot(matrixRL1,file_temp1)
#         matrixRL2 = 1 / (1 + np.exp(-matrix2))
#         all_action_val = np.dot(matrixRL2,file_temp2)
#         return all_action_val       
#     elif 170 < len(play_state) < 250  : #SHERIFF 
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = relu6_khanh_200922(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
#     else :#SUSHIGO-main, MACHIKOR0
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = np.tanh(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
    
# @njit()
# def neural_network_khanh_200922_2(play_state, file_temp0, file_temp1):    
#     if 170 < len(play_state) < 250  : #SHERIFF 
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = relu6_khanh_200922(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
#     else :#SUSHIGO-main, MACHIKOR0
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = np.tanh(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2

# @njit()
# def test2_Khanh_200922(play_state,file_per_2):
#     a = getValidActions(play_state)
#     a = np.where(a == 1)[0]
#     if len(file_per_2) == 3:
#         matran2 = neural_network_khanh_200922(play_state, file_per_2[0], file_per_2[1], file_per_2[2])
#     else:
#         matran2 = neural_network_khanh_200922_2(play_state, file_per_2[0], file_per_2[1])
#     max_ = 0
#     action_max = a[random.randrange(len(a))]
#     for act in a:
#         if matran2[act] > max_:
#             max_ = matran2[act]
#             action_max = act
#     return action_max

# #############################################################

# @njit()
# def neural_network_khanh_270922(play_state, file_temp0, file_temp1, file_temp2):
#     if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
#         matran1 = np.dot(play_state,file_temp0)
#         matran1 = 1 / (1 + np.exp(-matran1))
#         matran21 = np.dot(matran1,file_temp1)
#         matran21 *= (matran21 > 0)
#         matran2 = np.dot(matran21, file_temp2)
#         return matran2    
#     elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
#         matrix1 = np.dot(play_state,file_temp0)
#         matrixRL1 = np.tanh(matrix1)
#         matrix2 = np.dot(matrixRL1,file_temp1)
#         matrixRL2 = relu6_khanh_200922(matrix2)
#         all_action_val = np.dot(matrixRL2,file_temp2)
#         return all_action_val       
#     elif 170 < len(play_state) < 250  : #SHERIFF 
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = relu6_khanh_200922(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
#     else :#SUSHIGO-main, MACHIKOR0
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = np.tanh(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2

# @njit()
# def neural_network_khanh_270922_2(play_state, file_temp0, file_temp1):
#     if 170 < len(play_state) < 250  : #SHERIFF 
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = relu6_khanh_200922(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2
#     else :#SUSHIGO-main, MACHIKOR0
#         matran1 = np.dot(play_state, file_temp0)
#         matran1 = np.tanh(matran1)
#         matran2 = np.dot(matran1, file_temp1)
#         return matran2

# @njit()
# def test2_Khanh_270922(play_state,file_per_2):
#     a = getValidActions(play_state)
#     a = np.where(a == 1)[0]
#     if len(file_per_2) == 3:
#         matran2 = neural_network_khanh_270922(play_state, file_per_2[0], file_per_2[1], file_per_2[2])
#     else:
#         matran2 = neural_network_khanh_270922_2(play_state, file_per_2[0], file_per_2[1])
#     max_ = 0
#     action_max = a[random.randrange(len(a))]
    
#     for act in a:
#         if matran2[act] > max_:
#             max_ = matran2[act]
#             action_max = act
#     return action_max

# #################################################################
# #################################################################
# #################################################################
# @njit()
# def Identity_an_130922(x):
#     return x

# @njit()
# def BinaryStep_an_130922(x):
#     x[x>=0] = 1.0
#     x[x<0] = 0.0
#     return x

# @njit()
# def Sigmoid_an_130922(x):
#     return 1.0 / (1.0 + np.e**(-x))

# @njit()
# def NegativePositiveStep_an_130922(x):
#     x[x>=0] = 1.0
#     x[x<0] = -1.0
#     return x

# @njit()
# def Tanh_an_130922(x):
#     return (np.e**(x) - np.e**(-x)) / (np.e**(x) + np.e**(-x))

# @njit()
# def ReLU_an_130922(x):
#     return x * (x>0)

# @njit()
# def LeakyReLU_an_130922(x):
#     x[x<0] *= 0.01
#     return x

# @njit()
# def PReLU_an_130922(x, a=0.5):
#     x[x<0] *= 0.5
#     return x

# @njit()
# def Gaussian_an_130922(x):
#     return np.e**(-x**2)

# @njit()
# def id_function_an_130922(id, res_mat, Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922):
#     if id == 0: return Identity_an_130922(res_mat)
#     elif id == 1: return BinaryStep_an_130922(res_mat)
#     elif id == 2: return Sigmoid_an_130922(res_mat)
#     elif id == 3: return NegativePositiveStep_an_130922(res_mat)
#     elif id == 4: return Tanh_an_130922(res_mat)
#     elif id == 5: return ReLU_an_130922(res_mat)
#     elif id == 6: return LeakyReLU_an_130922(res_mat)
#     elif id == 7: return PReLU_an_130922(res_mat)
#     else: return Gaussian_an_130922(res_mat)

# # list_activation_function = [Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922]
# @njit()
# def neural_network_an_130922(res_mat, data, list_action):
#     for i in range(len(data)):
#         if i % 2 == 0:
#             res_mat = np.dot(res_mat, data[i])
#             max_x = np.max(np.abs(res_mat))
#             max_x_1 = max_x/25
#             res_mat = res_mat / max_x_1
#         else:
#             id = int(data[i][0][0])
#             # res_mat = list_activation_function[id](res_mat)
#             res_mat = id_function_an_130922(id, res_mat, Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922)

    
#     res_arr = res_mat[list_action]
#     arr_max = np.where(res_arr == np.max(res_arr))[0]
#     action_max_idx = np.random.choice(arr_max)
#     return list_action[action_max_idx]

# @njit()
# def test2_An_130922(p_state,  file_per_2):
#     list_action = getValidActions(p_state)
#     list_action = np.where(list_action == 1)[0]
#     action = neural_network_an_130922(p_state, file_per_2, list_action)
#     return action

# ############################################################
# @njit()
# def Identity_an_200922(x:np.ndarray):
#     return x/np.abs(x).max()

# @njit()
# def BinaryStep_an_200922(x:np.ndarray):
#     return np.where(x>=0, 1, 0).astype(np.float64)

# @njit()
# def Sigmoid_an_200922(x:np.ndarray):
#     LOG_INF = 709.782712893384
#     return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

# @njit()
# def SignStep_an_200922(x:np.ndarray):
#     return np.sign(x)

# @njit()
# def Tanh_an_200922(x:np.ndarray):
#     LOG_INF = 709.782712893384
#     HALF_LOG_INF = 354.891356446692
#     x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
#     return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

# @njit()
# def ReLU_an_200922(x:np.ndarray):
#     return np.where(x<0, 0, x)/np.max(x)

# @njit()
# def SoftPlus_an_200922(x:np.ndarray):
#     LOG_INF = 709.782712893384
#     x_ = np.where(np.abs(x)>LOG_INF-1, x, np.log(1+np.e**(x)))
#     return x_/np.max(x_)

# @njit()
# def Gaussian_an_200922(x:np.ndarray):
#     SQRT_LOG_INF = 18.838560360247595
#     return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

# @njit()
# def id_function_an_200922(id, res_mat, Identity_an_200922, BinaryStep_an_200922, Sigmoid_an_200922, SignStep_an_200922, Tanh_an_200922, ReLU_an_200922, SoftPlus_an_200922, Gaussian_an_200922):
#     if id == 0: return Identity_an_200922(res_mat)
#     elif id == 1: return BinaryStep_an_200922(res_mat)
#     elif id == 2: return Sigmoid_an_200922(res_mat)
#     elif id == 3: return SignStep_an_200922(res_mat)
#     elif id == 4: return Tanh_an_200922(res_mat)
#     elif id == 5: return ReLU_an_200922(res_mat)
#     elif id == 6: return SoftPlus_an_200922(res_mat)
#     else: return Gaussian_an_200922(res_mat)

# @njit()
# def Ann_neural_network_an_200922(res_mat:np.ndarray, data, list_action):
#     for i in range(len(data)//3):
#         data3i = data[3*i]
#         data3i1 = data[3*i+1].flatten()
#         data3i2 = int(data[3*i+2][0][0])
#         res_mat = np.dot(res_mat, data3i) + data3i1
#         res_mat = id_function_an_200922(data3i2, res_mat, Identity_an_200922, BinaryStep_an_200922, Sigmoid_an_200922, SignStep_an_200922, Tanh_an_200922, ReLU_an_200922, SoftPlus_an_200922, Gaussian_an_200922)
    
#     res_arr = res_mat[list_action]
#     a = np.max(res_arr)
#     if a >= 0:
#         arr_max = np.where(res_arr >= 0.99*a)[0]
#     else:
#         arr_max = np.where(res_arr >= 1.01*a)[0]

#     action_max_idx = np.random.choice(arr_max)
#     return list_action[action_max_idx]

# @njit()
# def test2_An_200922(p_state, file_per_2):
#     list_action = getValidActions(p_state)
#     list_action = np.where(list_action == 1)[0]
#     if len(file_per_2) == 2: 
#         type_file_per_2 = int(file_per_2[1][0][0][0])
#     else:
#         type_file_per_2 = int(file_per_2[0][1][0][0])
    
#     if type_file_per_2 == 0: # fnn
#         action = Ann_neural_network_an_200922(p_state, file_per_2[0], list_action)
#     else: # sg
#             res_arr = file_per_2[0][2][0][list_action]
#             a = np.max(res_arr)
#             if a >= 0:
#                 arr_max = np.where(res_arr >= 0.99*a)[0]
#             else:
#                 arr_max = np.where(res_arr >= 1.01*a)[0]
#             action_max_idx = np.random.choice(arr_max)
#             action = list_action[action_max_idx]

#     return action


# ####################################################

# @njit()
# def Identity_an_270922(x:np.ndarray):
#     return x/np.abs(x).max()

# @njit()
# def BinaryStep_an_270922(x:np.ndarray):
#     return np.where(x>=0, 1, 0).astype(np.float64)

# @njit()
# def Sigmoid_an_270922(x:np.ndarray):
#     LOG_INF = 709.782712893384
#     return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

# @njit()
# def SignStep_an_270922(x:np.ndarray):
#     return np.sign(x)

# @njit()
# def Tanh_an_270922(x:np.ndarray):
#     HALF_LOG_INF = 354.891356446692
#     x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
#     return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

# @njit()
# def ReLU_an_270922(x:np.ndarray):
#     return np.where(x<0, 0, x)/np.max(x)

# @njit()
# def LeakyReLU_an_270922(x:np.ndarray):
#     x_new = np.where(x<0, 0.01*x, x)
#     return x_new/np.abs(x_new).max()

# @njit()
# def PReLU_an_270922(x:np.ndarray):
#     x_new = np.where(x<0, 0.5*x, x)
#     return x_new/np.abs(x_new).max()

# @njit()
# def SoftPlus_an_270922(x:np.ndarray):
#     LOG_INF = 709.782712893384
#     x_new = np.where(np.abs(x)>LOG_INF-1e-9, x, np.log(1+np.e**(x)))
#     return x_new/np.max(x_new)

# @njit()
# def Gaussian_an_270922(x:np.ndarray):
#     SQRT_LOG_INF = 18.838560360247595
#     return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

# activation_function = [Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922]

# @njit()
# def id_function_an_270922(id, res_mat, Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922):
#     if id == 0: return Identity_an_270922(res_mat)
#     if id == 1: return BinaryStep_an_270922(res_mat)
#     if id == 2: return Sigmoid_an_270922(res_mat)
#     if id == 3: return SignStep_an_270922(res_mat)
#     if id == 4: return Tanh_an_270922(res_mat)
#     if id == 5: return ReLU_an_270922(res_mat)
#     if id == 6: return LeakyReLU_an_270922(res_mat)
#     if id == 7: return PReLU_an_270922(res_mat)
#     if id == 8: return SoftPlus_an_270922(res_mat)
#     else: return Gaussian_an_270922(res_mat)

# @njit()
# def Ann_neural_network_an_270922(res_mat:np.ndarray, data, list_action):
#     for i in range(len(data)//3):
#         data3i = data[3*i]
#         data3i1 = data[3*i+1].flatten()
#         data3i2 = int(data[3*i+2][0][0])
#         res_mat = np.dot(res_mat, data3i) + data3i1
#         res_mat = id_function_an_270922(data3i2, res_mat, Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922)
    
#     res_arr = res_mat[list_action]
#     a = np.max(res_arr)
#     if a >= 0:
#         arr_max = np.where(res_arr >= 0.99*a)[0]
#     else:
#         arr_max = np.where(res_arr >= 1.01*a)[0]
    
#     return list_action[np.random.choice(arr_max)]

# @njit()
# def test2_An_270922(p_state, file_per_2):
#     list_action = getValidActions(p_state)
#     list_action = np.where(list_action == 1)[0]
#     if len(file_per_2) == 2: 
#         type_file_per_2 = int(file_per_2[1][0][0][0])
#     else:
#         type_file_per_2 = int(file_per_2[0][1][0][0])
#     if type_file_per_2 == 0:
#         action = Ann_neural_network_an_270922(p_state, file_per_2[0], list_action)
#     else:
#         res_arr = file_per_2[0][2][0][list_action]
#         a = np.max(res_arr)
#         arr_max = np.where(res_arr >= 0.99*a)[0]
#         action = list_action[np.random.choice(arr_max)]
        
#     return action


# #################################################################
# #################################################################
# #################################################################

# @njit()
# def test2_Dat_130922(state,file_per_2):
#     list_action = getValidActions(state)
#     list_action = np.where(list_action == 1)[0]
#     hidden1 = np.dot(state, file_per_2[0])
#     hidden2 = hidden1 * (hidden1>0)
#     values =  np.dot(hidden2, file_per_2[1])
#     action = list_action[np.argmax(values[list_action])]
#     return action


# ###############################################################
# ###############################################################
# ###############################################################
# @njit()
# def neural_network_hieu_130922(state, file_temp0, file_temp1, file_temp2, list_action):
#     norm_state = state/np.linalg.norm(state, 1)
#     norm_state = np.tanh(norm_state)                    #d???ng tanh
#     norm_action = np.zeros(getActionSize())
#     norm_action[list_action] = 1
#     norm_action = norm_action.reshape(1, getActionSize())
#     matrixRL1 = np.dot(norm_state, file_temp0)
#     matrixRL1 = matrixRL1*(matrixRL1 > 0)           #activation = relu
#     matrixRL2 = np.dot(matrixRL1, file_temp1)
#     matrixRL2 = 1 / (1 + np.exp(-matrixRL2))            #activation = sigmoid
#     matrixRL3 = np.dot(matrixRL2, file_temp2)
#     matrixRL3 = np.tanh(matrixRL3)              #activation = tanh
#     result_val_action = matrixRL3*norm_action
#     action_max = np.argmax(result_val_action)
#     return action_max

# @njit()
# def test2_Hieu_130922(state, file_per_2):
#     list_action = getValidActions(state)
#     list_action = np.where(list_action == 1)[0]
#     action = neural_network_hieu_130922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
#     return action
# #################################################################
# @njit()
# def agent_hieu_270922(state,file_temp,file_per):
#     actions = getValidActions(state)
#     actions = np.where(actions == 1)[0]
#     action = np.random.choice(actions)
#     file_per = (len(state),getActionSize())
#     return action,file_temp,file_per

# # LEN_STATE_hieu_270922,AMOUNT_ACTION_hieu_270922 = normal_main([agent_hieu_270922]*getAgentSize(), 1, [0])[1]

# @njit()
# def softmax_hieu_270922(X):
#     expo = np.exp(X)
#     return expo/np.sum(expo)

# @njit()
# def sigmoid_hieu_270922(X):
#     return 1/(1+np.exp(-X))

# @njit()
# def tanh_hieu_270922(X):
#     return np.tanh(X)

# @njit()
# def neural_network_hieu_270922(norm_state, file_temp0, file_temp1, file_temp2, list_action):
#     # norm_state = state.copy()
#     norm_state = norm_state/np.linalg.norm(norm_state, 1)
#     norm_state = softmax_hieu_270922(norm_state)
#     norm_action = np.zeros(getActionSize())
#     norm_action[list_action] = 1
#     norm_action = norm_action.reshape(1, getActionSize())

#     matrixRL1 = np.dot(norm_state, file_temp0)
#     matrixRL1 = sigmoid_hieu_270922(matrixRL1)          

#     matrixRL2 = np.dot(matrixRL1, file_temp1)
#     matrixRL2 = tanh_hieu_270922(matrixRL2)         

#     matrixRL3 = np.dot(matrixRL2, file_temp2)
#     matrixRL3 = softmax_hieu_270922(matrixRL3)   

#     result_val_action = matrixRL3*norm_action
#     action_max = np.argmax(result_val_action)
#     return action_max

# @njit()
# def test2_Hieu_270922(state, file_per_2):
#     list_action = getValidActions(state)
#     list_action = np.where(list_action == 1)[0]
#     action = neural_network_hieu_270922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
#     return action

# ######################################################################
# ######################################################################
# ######################################################################

# @njit()
# def file_temp_to_action_Phong_130922(state, file_temp):
#     a = getValidActions(state)
#     a = np.where(a == 1)[0]
#     RELU = np.ones(len(state))
#     matrix_new = np.dot(RELU,file_temp)
#     list_val_action = matrix_new[a]
#     action = a[np.argmax(list_val_action)]
#     return action

# @njit() 
# def test2_Phong_130922(state,file_per_2):
#     action = file_temp_to_action_Phong_130922(state, file_per_2)
#     return action













# @njit()
# def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
#     if id == 0: return test2_An_130922(player_state, per0)
#     elif id == 1: return test2_Dat_130922(player_state, per1)
#     elif id == 2: return test2_Hieu_130922(player_state, per2)
#     elif id == 3: return test2_Hieu_270922(player_state, per3)
#     elif id == 4: return test2_Khanh_200922(player_state, per4)
#     elif id == 5: return test2_NhatAnh_130922(player_state, per5)
#     elif id == 6: return test2_NhatAnh_200922(player_state, per6)
#     elif id == 7: return test2_NhatAnh_270922(player_state, per7)
#     elif id == 8: return test2_Phong_130922(player_state, per8)
#     elif id == 9: return test2_Phong_130922(player_state, per9)
#     else: return test2_Phong_130922(player_state, per10)


# @njit()
# def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
#     env = initEnv()


#     winner = -1
#     while env[230] < MAX_TURN_IN_ONE_GAME:
#         idx = int(env[254])
#         player_state = getAgentState(env)
#         if list_other[idx] == -1:
#             action, per_player = p0(player_state,per_player)
#         else:
#             action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
#         if getValidActions(player_state)[action] != 1:
#             raise Exception('bot dua ra action khong hop le')
#         stepEnv(env, action)

#         winner = checkEnded(env)
#         if winner != -1:
#             break

#     env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]
#     if winner != -1:
#         for i in range(4):
#             env[254] = i
#             env[229] = 2
#             p_state = getAgentState(env)
#             if list_other[i] == -1:
#                 act, per_player = p0(p_state, per_player)

#     winner_ = False
#     if np.where(list_other == -1)[0] ==  winner: winner_ = True
#     else: winner_ = False
#     return winner_,  per_player



# @njit()
# def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
#     win = 0
#     for _n in range(num_game):
#         list_other = np.append(np.random.choice(np.arange(11), 3), -1)
#         np.random.shuffle(list_other)
#         winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
#         win += winner
#     return win, per_player

# def numba_main_2(p0, per_player, n_game):
#     list_all_players = dict_game_for_player[game_name_]
#     list_data = load_data_per2(list_all_players, game_name_)
#     per0 = list_data[0]
#     per1 = list_data[1]
#     per2 = list_data[2]
#     per3 = list_data[3]
#     per4 = list_data[4]
#     per5 = list_data[5]
#     per6 = list_data[6]
#     per7 = list_data[7]
#     per8 = list_data[8]
#     per9 = list_data[9]
#     per10 = list_data[10]
#     return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)

# @njit()
# def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
#     env = initEnv()


#     winner = -1
#     while env[230] < MAX_TURN_IN_ONE_GAME:
#         idx = int(env[254])
#         player_state = getAgentState(env)
#         if list_other[idx] == -1:
#             action, per_player = p0(player_state,per_player)
#         else:
#             action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
#         if getValidActions(player_state)[action] != 1:
#             raise Exception('bot dua ra action khong hop le')
#         stepEnv(env, action)

#         winner = checkEnded(env)
#         if winner != -1:
#             break

#     env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]
#     if winner != -1:
#         for i in range(4):
#             env[254] = i
#             env[229] = 2
#             p_state = getAgentState(env)
#             if list_other[i] == -1:
#                 act, per_player = p0(p_state, per_player)

#     winner_ = False
#     if np.where(list_other == -1)[0] ==  winner: winner_ = True
#     else: winner_ = False
#     return winner_,  per_player



# @njit()
# def n_game_numba_2(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
#     win = 0
#     for _n in range(num_game):
#         list_other = np.append(np.random.choice(np.arange(11), 3), -1)
#         np.random.shuffle(list_other)
#         winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
#         win += winner
#     return win, per_player

# def normal_main_2(p0, per_player, n_game):
#     list_all_players = dict_game_for_player[game_name_]
#     list_data = load_data_per2(list_all_players, game_name_)
#     per0 = list_data[0]
#     per1 = list_data[1]
#     per2 = list_data[2]
#     per3 = list_data[3]
#     per4 = list_data[4]
#     per5 = list_data[5]
#     per6 = list_data[6]
#     per7 = list_data[7]
#     per8 = list_data[8]
#     per9 = list_data[9]
#     per10 = list_data[10]
#     return n_game_numba_2(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)

