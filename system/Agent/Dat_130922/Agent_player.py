import numpy as np
from env import *
import random
import os
import sys
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))

time_run_game = 79200
player = 'Dat_130922'  #Tên folder của người chơi
path_data = f'system/Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def agent(state,file_temp,file_per):
    action = np.random.choice(get_list_action(state))
    file_per = (len(state),amount_action())
    return action,file_temp,file_per

LEN_STATE,AMOUNT_ACTION = normal_main([agent]*amount_player(), 1, [0])[1]
LEN_STATE,AMOUNT_ACTION
def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
def sigmoid(x):
    return 1 / (1+np.exp(-x))
class GeneticAlgorithm:
    def __init__(self):
        try:
            self.best_agent = self.Agent()
            self.best_agent2 = self.Agent()
            data1 = np.load(os.path.join(path_save_player,'Best_Matrix.npz'))
            data2 = np.load(os.path.join(path_save_player,'Best_Matrix2.npz'))
            data3 = np.load(os.path.join(path_save_player,'Best.npz'))
            self.best_agent.weights1,self.best_agent.weights2 = data1['w1'],data1['w2']
            self.best_agent2.weights1,self.best_agent2.weights2 = data2['w1'],data2['w2']
            self.best.weights1,self.best.weights2 = data3['w1'],data3['w2']
        except:
            self.best_agent = 0
            self.best_agent2 = 0
            self.best = 0
        self.list_agent = []
    class Agent:
        """Agent with random weights"""
        def __init__(self):
            if np.random.rand() >= 0.5:
                self.weights1 = np.random.rand(LEN_STATE,256)*3 - 1.5
                self.weights2 = np.random.rand(256,AMOUNT_ACTION)*2-1
            else:
                self.weights1 = np.random.choice(np.arange(-1.5,1.5,0.25),size=(LEN_STATE,256))
                self.weights2 = np.random.choice(np.arange(-1,1,0.25),size=(256,AMOUNT_ACTION))

        def play(self,state,temp,per):
            list_action = get_list_action(state)
            hidden1 = state.dot(self.weights1)
            hidden2 = hidden1 * (hidden1>0)
            values = hidden2.dot(self.weights2)
            action = list_action[np.argmax(values[list_action])]
            return action,temp,per

    def UniformCrossover(self,Agent1,Agent2):
        """Return a child Agent with crossover weights between two parents Agent"""
        """Uniform Crossover"""
        ChildAgent = self.Agent()
        choice1 = np.random.randint(2, size = Agent1.weights1.shape).astype(bool)
        choice2 = np.random.randint(2, size = Agent1.weights2.shape).astype(bool)
        ChildAgent.weights1,ChildAgent.weights2 =  np.where(choice1,Agent1.weights1,Agent2.weights1),np.where(choice2,Agent1.weights2,Agent2.weights2)
        return ChildAgent
    
    def LinearCrossover(self,Agent1,Agent2):
        ChildAgent = self.Agent()
        ChildAgent = self.Agent()
        choice1 = np.random.randint(2, size = Agent1.weights1.shape).astype(bool)
        choice2 = np.random.randint(2, size = Agent1.weights2.shape).astype(bool)
        if np.random.rand()>=0.5:
            ChildAgent.weights1 = np.where(choice1,Agent1.weights1 * 0.5 + Agent2.weights1 * 0.5,Agent2.weights1)
        else:
            ChildAgent.weights1 = np.where(choice1,Agent1.weights1 * 0.5 + Agent2.weights1 * 0.5,Agent1.weights1)
        if np.random.rand()>=0.5:
            ChildAgent.weights2 = np.where(choice2,Agent1.weights2 * 0.5 + Agent2.weights2 * 0.5,Agent2.weights2)
        else:
            ChildAgent.weights2 = np.where(choice2,Agent1.weights2 * 0.5 + Agent2.weights2 * 0.5,Agent1.weights2)
        return ChildAgent
    def Mutation(self,Agent1,rate=0.01):
        """Small chance of changing some weights of child Agent"""
        Dummy = self.Agent()
        choice1 = np.random.choice([0,1],size=Dummy.weights1.shape,p=[rate,1-rate]).astype(bool)
        choice2 = np.random.choice([0,1],size=Dummy.weights2.shape,p=[rate,1-rate]).astype(bool)
        Agent1.weights1,Agent1.weights2 =  np.where(choice1,Agent1.weights1,Dummy.weights1),np.where(choice2,Agent1.weights2,Dummy.weights2)
        return Agent1
    def ChildAgent(self,Agent1,Agent2,mutation_rate = 0.1):
        """Create Agent using Crossover and Mutation"""
        if np.random.rand() >0.4:
            Child = self.UniformCrossover(Agent1,Agent2)
        else:
            Child = self.LinearCrossover(Agent1,Agent2)
        if np.random.rand() > mutation_rate:
            return Child
        else:
            return self.Mutation(Child)
    
    def agent_random(self,state,temp,per):
        """Agent random."""
        list_action = get_list_action(state)
        idx = random.randrange(len(list_action))
        return list_action[idx],temp,per

    def choose_better_Agent(self,Agent1,Agent2) -> Agent:
        list_player = [Agent1.play,Agent2.play] + [self.agent_random] * (amount_player()-2)
        rank = np.array(normal_main(list_player, 150, [0])[0])[:2] + np.array(normal_main(list_player, 150, [0])[0])[:2]
        if np.argmax(rank)==0:
            return Agent1
        elif np.argmax(rank)==1:
            return Agent2
        else:
            pass
    def ChooseAgent(self,list_agent):
        idx_a  =random.randrange(len(list_agent))
        idx_b = random.randrange(len(list_agent))
        while idx_a==idx_b:
            idx_b = random.randrange(len(list_agent))
        Agent1,Agent2 = list_agent[idx_a],list_agent[idx_b]
        return Agent1,Agent2
    def Generation0(self) -> Agent:
        """Return best random Agent to be Gen 0"""
        list_agent = [self.Agent() for _ in range(128)]
        #print('Choosing best random:',end=" ")
        while len(list_agent)>1:
                Agent1, Agent2 = self.ChooseAgent(list_agent)
                agent = self.choose_better_Agent(Agent1,Agent2)
                if agent == Agent1:
                    list_agent.remove(Agent2)
                elif agent==Agent2:
                    list_agent.remove(Agent1)
                else:
                    list_agent.remove(Agent1)
                    list_agent.remove(Agent2)
                #print(len(list_agent),end = " ")
        else:
            pass
        #print("Gen0 created!")
        best_agent = list_agent[0]
        return best_agent
    def NextGeneration(self,*Agent):
        for i in range(self.num_agents-2):
            Agent1,Agent2 = np.random.choice(Agent,size=2,replace=False)
            self.list_agent.append(self.ChildAgent(Agent1,Agent2))
        ##print("Next Gen Created!")
    def BestAgent(self,list_agent_) -> Agent:
        """Return best agent in this generation"""
        list_agent = list_agent_.copy()
        ##print("Choosing Best Agent:",end=" ")
        while len(list_agent)>1:
                Agent1, Agent2 = self.ChooseAgent(list_agent)
                agent = self.choose_better_Agent(Agent1,Agent2)
                # ##print(agent)
                if agent == Agent1:
                    list_agent.remove(Agent2)
                elif agent == Agent2:
                    list_agent.remove(Agent1)
                ##print(len(list_agent),end=" ")
        else:
            pass
        best_agent = list_agent[0]
        
        ##print("Best Agent has been chosen!")
        return best_agent
        
    def reset(self):
        self.list_agent = []

    def train(self,num_generations = 10,num_agents = 64):
        self.num_agents = num_agents
        for num in range(num_generations):
            if num==0 and self.best_agent==0 and self.best_agent2==0:
                self.list_agent.append(self.Generation0())
                self.list_agent.append(self.Generation0())
                self.NextGeneration(self.list_agent[0],self.list_agent[1])
                self.best_agent = self.BestAgent(self.list_agent)
                self.list_agent.remove(self.best_agent)
                self.best_agent2 = self.BestAgent(self.list_agent)
                self.reset()
                lst1 = [self.best_agent.play,self.best_agent2.play]+ [self.agent_random]*(amount_player()-2)
                rank = normal_main(lst1,5000,[0])[0][:2]
                if rank[1]>rank[0]+50:
                    self.best_agent,self.best_agent2 = self.best_agent2,self.best_agent
                else:
                    pass
                try:
                    lst2 = [self.best.play,self.best_agent.play]+ [self.agent_random]*(amount_player()-2)
                    rank2 = normal_main(lst2,10000,[0])[0][:2]
                    if rank2[1]>rank2[0]+100:
                        self.best = self.best_agent
                    else:
                        pass
                except:
                    self.best = self.best_agent
                np.savez(os.path.join(path_save_player,'Best_Matrix.npz'),w1=self.best_agent.weights1,w2=self.best_agent.weights2)
                np.savez(os.path.join(path_save_player,'Best_Matrix2.npz'),w1=self.best_agent2.weights1,w2=self.best_agent2.weights2)
                np.savez(os.path.join(path_save_player,'Best.npz'),w1=self.best.weights1,w2=self.best.weights2)
                self.reset()
            else:
                if np.random.rand()<0.4:
                    best_random = self.Generation0()
                    self.list_agent.append(best_random)
                    self.list_agent.append(self.best_agent)
                    self.list_agent.append(self.best_agent2)
                    self.NextGeneration(self.best_agent,self.best_agent2,best_random)
                else:
                    self.list_agent.append(self.best_agent)
                    self.list_agent.append(self.best_agent2)
                    self.NextGeneration(self.best_agent,self.best_agent2)
                self.best_agent = self.BestAgent(self.list_agent)
                self.list_agent.remove(self.best_agent)
                self.best_agent2 = self.BestAgent(self.list_agent)
                self.reset()
                lst1 = [self.best_agent.play,self.best_agent2.play]+ [self.agent_random]*(amount_player()-2)
                rank = normal_main(lst1,5000,[0])[0][:2]
                if rank[1]>=rank[0]+50:
                    self.best_agent,self.best_agent2 = self.best_agent2,self.best_agent
                else:
                    pass
                lst2 = [self.best.play,self.best_agent.play]+ [self.agent_random]*(amount_player()-2)
                rank2 = normal_main(lst2,10000,[0])[0][:2]
                if rank2[1]>=rank2[0]+100:
                    self.best = self.best_agent
                else:
                    pass
                np.savez(os.path.join(path_save_player,'Best_Matrix.npz'),w1=self.best_agent.weights1,w2=self.best_agent.weights2)
                np.savez(os.path.join(path_save_player,'Best_Matrix2.npz'),w1=self.best_agent2.weights1,w2=self.best_agent2.weights2)
                np.savez(os.path.join(path_save_player,'Best.npz'),w1=self.best.weights1,w2=self.best.weights2)
            print(num)



model = GeneticAlgorithm()
def train(num_gen=10,num_agents = 64):
    model.train(num_generations=num_gen,num_agents=num_agents)

def test(state,temp,per):
    if len(temp)<2:
        player = 'Dat_130922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        data = np.load(os.path.join(path_save_player,'Best.npz'))
        temp = [data['w1'],data['w2']]

    list_action = get_list_action(state)
    hidden1 = state.dot(temp[0])
    hidden2 = hidden1 * (hidden1>0)
    values = hidden2.dot(temp[1])
    action = list_action[np.argmax(values[list_action])]
    return action,temp,per
    
    


