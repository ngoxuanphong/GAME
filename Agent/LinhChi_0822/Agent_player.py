import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
import numpy as np
import pandas as pd
import random

player = 'LinhChi_0822'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


def random_p(state,file_temp,file_per):
    list_action = get_list_action(state)
    action = np.random.choice(list_action)
    return action,file_temp,file_per


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.idx = 0

    def store(self, state, action, next_state):
        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state

        self.idx = (self.idx+1)%self.capacity

    def sample(self, batch_size):
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)
        states = np.array(self.buffer_state)[indices_to_sample]
        actions = np.array(self.buffer_action)[indices_to_sample]
        next_states = np.array(self.buffer_next_state)[indices_to_sample]
        return states, actions, next_states


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers  # layer [1, 256, amount_action]
        self.alpha = alpha  # learning rate

        self.W = []
        self.b = []
        # Khởi tạo các tham số ở mỗi layer
        for i in range(0, len(layers) - 1):
            w_ = 2 * np.random.randn(layers[i], layers[i + 1]) - 1
            b_ = np.zeros((layers[i + 1], 1))
            self.W.append(w_ / layers[i])
            self.b.append(b_)

    # Train mô hình với dữ liệu
    def fit_partial(self, state, action, next_state, posible_actions):
        x = np.copy(state)
        x = x.reshape(-1, 1)
        A = [x]

        # feedforward online network
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        # lấy ra Q-value của action
        y = A[-1][:, action]

        # feedforward target network
        # if check_victory(next_state):
        #     y1 = np.copy(y)
        # else:
        x1 = np.copy(next_state)
        x1 = x1.reshape(-1, 1)
        A1 = [x1]
        out = A1[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A1.append(out)
        sum_eval = np.sum(A1[-1], axis=0)
        # lấy ra Q-value của action
        max = -np.inf
        action_max = 0
        for a in posible_actions:
            if sum_eval[a] > max:
                max = sum_eval[a]
                action_max = a
        y1 = A1[-1][:, a]

        # quá trình backpropagation
        y = y.reshape(-1, 1)
        dA = [-(y / A[-1] - (1 - y) / (1 - A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers) - 1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i + 1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i + 1]), 0)).reshape(-1, 1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i + 1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
        # Đảo ngược dW, db
        dW = dW[::-1]
        db = db[::-1]

        # Gradient descent
        for i in range(0, len(self.layers) - 1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]


def Chi(state, file_temp, file_per):
    history = ReplayMemory(10000)
    train_batch_size = 64
    actions = get_list_action(state)
    action = random.choice(actions)
    layer = [1, 128, amount_action()]

    if len(file_per) < 2:
        file_per = [[], [], [], [], []]
        # model
        file_per[0] = NeuralNetwork(layer)

    if len(file_temp) < 2:
        file_temp = [[], [], [], [], []]
        '''
        0: state
        1: action
        2: W
        3: B
        '''

        file_temp[2] = file_per[0].W
        file_temp[3] = file_per[0].b
    file_temp[0].append(state)
    if len(history.buffer_state) > train_batch_size:
        x1 = np.copy(state)
        x1 = x1.reshape(-1, 1)
        A1 = [x1]
        out = A1[-1]
        for i in range(0, len(layer) - 1):
            out = sigmoid(np.dot(out, file_temp[2][i]) + (file_temp[3][i].T))
            A1.append(out)
        sum_e = np.sum(A1[-1], axis=0)
        max = -np.inf
        action_max = 0
        for a in actions:
            if sum_e[a] > max:
                max = sum_e[a]
                action_max = a
        action = action_max

    file_temp[1].append(action)
    if check_victory(state) == 1:
        for i in range(len(file_temp[0]) - 1):
            history.store(file_temp[0][i], file_temp[1][i], file_temp[0][i + 1])
    if check_victory(state) != -1:
        if len(history.buffer_state) > train_batch_size:
            states, actions, next_states = history.sample(train_batch_size)
            posible_actions = [get_list_action(s) for s in next_states]
            for i in range(train_batch_size):
                for _ in range(5):
                    file_per[0].fit_partial(states[i], actions[i], next_states[i], posible_actions[i])

    return action, file_temp, file_per


def test(state,file_temp,file_per):
    history = ReplayMemory(10000)
    train_batch_size = 64
    actions = get_list_action(state)
    action = random.choice(actions)
    layer = [1, 128, amount_action()]

    player = 'LinhChi_0822'
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'

    if len(file_per) < 2:
        file_per = [[], [], [], [], []]
        # model
        file_per[0] = np.load(f'{path_save_player}Matrix.npy',allow_pickle=True)
        file_per[1] = np.load(f'{path_save_player}Bias.npy',allow_pickle=True)

    if len(file_temp) < 2:
        file_temp = [[], [], [], [], []]
        '''
        0: state
        1: action
        2: W
        3: B
        '''
        file_temp[2] = file_per[0]
        file_temp[3] = file_per[1]
    file_temp[0].append(state)
    x1 = np.copy(state)
    x1 = x1.reshape(-1, 1)
    A1 = [x1]
    out = A1[-1]
    for i in range(0, len(layer) - 1):
        out = sigmoid(np.dot(out, file_temp[2][i]) + (file_temp[3][i].T))
        A1.append(out)
    sum_e = np.sum(A1[-1], axis=0)
    max = -np.inf
    action_max = 0
    for a in actions:
        if sum_e[a] > max:
            max = sum_e[a]
            action_max = a
    action = action_max

    file_temp[1].append(action)
    if check_victory(state) == 1:
        for i in range(len(file_temp[0])-1):
            history.store(file_temp[0][i], file_temp[1][i], file_temp[0][i+1])
    if check_victory(state) != -1:
        if len(history.buffer_state) > train_batch_size:
            states, actions, next_states = history.sample(train_batch_size)
            posible_actions = [get_list_action(s) for s in next_states]
            for i in range(train_batch_size):
                for _ in range(5):
                    file_per[0].fit_partial(states[i], actions[i], next_states[i], posible_actions[i])

    return action, file_temp, file_per


def train(n):
    history = ReplayMemory(10000)
    list_player= [random_p]*amount_player()
    list_player[0] = Chi
    kq, file_ = normal_main(list_player,100,[0])
    np.save(f'{path_save_player}Matrix.npy',file_[0].W)
    np.save(f'{path_save_player}Bias.npy',file_[0].b)
    for i in range(1000000):
        list_player= [random_p]*amount_player()
        list_player[0] = test
        kq, file_ = normal_main(list_player,50,[0])
        np.save(f'{path_save_player}Matrix.npy',file_[0].W)
        np.save(f'{path_save_player}Bias.npy',file_[0].b)




