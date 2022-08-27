from torch.autograd import Variable
from env import *
import torch
import torch.nn as nn
import numpy
import numpy as np
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))

player = 'ĐìnhĐạt_0822'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


def agent(state,file_temp,file_per):
    action = np.random.choice(get_list_action(state))
    file_per = (len(state),amount_action())
    return action,file_temp,file_per
if amount_player()==4:
    LEN_STATE,AMOUNT_ACTION = normal_main([agent,agent,agent,agent], 1, [0])[1]
else:
    LEN_STATE,AMOUNT_ACTION = normal_main([agent,agent,agent,agent,agent], 1, [0])[1]
# print(LEN_STATE,AMOUNT_ACTION)


class ActorNetwork(nn.Module):
    def __init__(self,n_inputs,n_outputs):
        super(ActorNetwork,self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(n_inputs,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,n_outputs),
            nn.Softmax(dim=-1)
        )
    def forward(self,state):
        state = torch.as_tensor(state,dtype=torch.float) / 128
        policy = self.actor_net(state)
        return policy

class CriticNetwork(nn.Module):
    def __init__(self,n_inputs,n_outputs):
        super(CriticNetwork,self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(n_inputs,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,n_outputs)
        )
    def forward(self,state):
        state = torch.as_tensor(state,dtype=torch.float) / 128
        value = self.critic_net(state)
        return value
class PPOModel:
    def __init__(self,lr = 25e-7,gamma=0.99,gae_lambda=0.95,policy_clip=0.2):
        self.len_state = LEN_STATE
        self.len_action_space = AMOUNT_ACTION
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.actor = ActorNetwork(self.len_state,self.len_action_space) #policy network
        self.critic = CriticNetwork(self.len_state,1)#value network
        self.actor_old = ActorNetwork(self.len_state,self.len_action_space)# old policy net work
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.state = [] #store states each game
        self.action = [] #store all action each game
        self.prob = [] #store probability each action each time step each game
        self.reward = [] #store reward each game
        self.reward_G = []
        self.values = []
        self.len_eps = []
        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': self.lr},
                        {'params': self.critic.parameters(), 'lr': self.lr}
                    ])
        self.victory = []
    
    def reset_params(self):
        self.state = []
        self.action = []
        self.prob = []
        self.reward = []
        self.values = []
        self.reward_G = []
    def update_params(self,agent_A,agent_B,agent_C,agent_D,num):
        data = self.play(agent_A,agent_B,agent_C,agent_D)
        if data[3]==0:
            self.victory.append(0)
        else:
            self.victory = []
        # print(data[3],end=" ")
        self.state = torch.tensor(np.array(data[0]),dtype=torch.float)
        self.action = torch.tensor(np.array(data[1]),dtype=torch.float)
        self.prob = torch.tensor(np.array(data[2]),dtype=torch.float)
        self.reward = torch.zeros_like(self.action)
        self.len_eps.append(len(self.reward))
        self.reward[-1]+=(data[3]) * (min(self.len_eps)) ** 2 / len(self.reward)**2
        # print(len(self.reward))
        # print(self.reward)
        discounted_reward = 0
        for rew in reversed(self.reward):
            discounted_reward = rew + discounted_reward * self.gamma
            self.reward_G.append(discounted_reward)
        
        self.reward_G = torch.tensor(np.array(self.reward_G),dtype=torch.float)
        self.reward_G = reversed(self.reward_G)
        # print(self.reward_G)
        # print(len(self.reward))
    def agent_random(self,state,file_temp,file_per):
        action = np.random.choice(get_list_action(state))
        return action,file_temp,file_per
    def agent_Dat_random(self,state,file_temp,file_per):
        if len(file_per) < 2:
            file_per = [[],[],[],0] # file_per contain [state,action,probability,reward]
        list_action = get_list_action(state)
        prob = self.actor.forward(state)
        prob = prob.detach().numpy()
        # print(prob)
        action = np.random.choice(list_action)
        file_per[0].append(state)
        file_per[1].append(action)
        file_per[2].append(list(prob))

        if check_victory(state)!=-1:
            file_per[3] = check_victory(state)
        return action, file_temp,file_per
    
    def agent_Dat(self,state,file_temp,file_per):
        if len(file_per) < 2:
            file_per = [[],[],[],0] # file_per contain [state,action,probability,reward]
        list_action = get_list_action(state)
        prob = self.actor.forward(state)
        prob = prob.detach().numpy()
        # print(prob)
        action = list_action[np.argmax(prob[list_action])]
        file_per[0].append(state)
        file_per[1].append(action)
        file_per[2].append(list(prob))

        if check_victory(state)!=-1:
            file_per[3] = check_victory(state)
        return action, file_temp,file_per
        
    def agent_train(self,state,file_temp,file_per):
        if len(file_temp)<2:
            file_temp = [0,[[],[],[],[],0],[0]]
            try:
                model = torch.load('actor_net.pt')
                model.eval()
                file_temp[0] = model
            except:
                file_temp[0] = self.actor
        list_action = get_list_action(state)
        prob = file_temp[0].forward(state).detach().numpy()
        action = list_action[np.argmax(prob[list_action])]
        return action,file_temp,file_per


    def play(self,agent_A,agent_B,agent_C,agent_D):
        list_player = [agent_A,agent_B,agent_C,agent_D]
        if amount_player() >4:
            list_player.append(self.agent_random)
        _,data =  normal_main(list_player, 1, [0])
        return data
    
    def check_reset_model(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        if len(self.victory) > 200:
            self.actor.apply(weight_reset)
            self.critic.apply(weight_reset)
            # print('All model reset! (Bad init weight)')
    def train(self,num_games,train_with_random=True):
        for num in range(num_games+1):
            #get data
            if num<1500 and train_with_random==True:
                self.update_params(self.agent_Dat_random,self.agent_random,self.agent_random,self.agent_random,num)
            else:
                self.update_params(self.agent_Dat,self.agent_train,self.agent_random,self.agent_random,num)
            self.check_reset_model()
            for step in range(len(self.action)):
                #calculate the prob_ratios
                value = self.critic.forward(self.state[step]).squeeze()
                advantage = self.reward_G[step] - value.detach()
                for _ in range(1):
                    current_prob = self.actor.forward(self.state[step])
                    # dist = torch.distributions.Categorical(current_prob)
                    act_prob = current_prob[int(self.action[step].item())] 
                    old_act_prob = self.prob[step][int(self.action[step].item())].detach()
                    prob_ratios = act_prob / old_act_prob
                    #calculate surrogate loss
                    weighted_prob = prob_ratios * advantage
                    weighted_clipped_prob = torch.clamp(prob_ratios,1-self.policy_clip,1+self.policy_clip) * advantage
                    #calculate actor loss
                    actor_loss = -1 * (torch.min(weighted_prob,weighted_clipped_prob)).mean()

                    critic_loss = nn.MSELoss()(value,self.reward_G[step])
                    total_loss = actor_loss + 0.5 * critic_loss #- 0.01 * dist_entropy
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
            self.reset_params()
            if num%50==0:
                # print(f'Game:{num}')
                if num%250==0:
                    torch.save(self.actor,os.path.join(path_save_player,'actor_net.pt'))
                    torch.save(self.critic,os.path.join(path_save_player,'critic_net.pt'))
                else:
                    pass
        torch.save(self.actor,os.path.join(path_save_player,'actor_net.pt'))
        torch.save(self.critic,os.path.join(path_save_player,'critic_net.pt'))
        # print("Model Saved!")
        


model = PPOModel()
def test(p_state, temp_file, per_file):
            player = 'ĐìnhĐạt_0822'
            path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
            # print(path_save_player)
            if per_file==[0]:
                model = torch.load(os.path.join(path_save_player,'actor_net.pt'))
                model.eval()
                per_file = model
            list_action = get_list_action(p_state)
            # print(per_file)
            prob = per_file.forward(p_state)
            prob = prob.detach().numpy()
            # print(prob)
            action = list_action[np.argmax(prob[list_action])]
            return action, temp_file, per_file

def train(num_games):
    model.train(num_games=int(num_games*10000),train_with_random=True)

