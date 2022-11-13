import numpy as np
from env import *
import random
import json
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))

player = 'Dat_270922'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)
    
def agent(state,file_temp,file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    file_per = (len(state),getActionSize())
    return action,file_temp,file_per

LEN_STATE,AMOUNT_ACTION = normal_main([agent]*getAgentSize(), 1, [0])[1]
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X),keepdims=True)
class AgentPolicy:
    def __init__(self,num_game):
        self.count = np.ones(AMOUNT_ACTION)
        self.reward = np.ones(AMOUNT_ACTION)
        self.policy = np.zeros(AMOUNT_ACTION)
        self.num_games = num_game
    def agent_random(self,state,temp,per):
        """Agent random."""
        list_action = getValidActions(state)
        list_action = np.where(list_action == 1)[0]
        idx = random.randrange(len(list_action))
        return list_action[idx],temp,per
    def run(self,state,temp,per):
        list_action = getValidActions(state)
        list_action = np.where(list_action == 1)[0]
        action = list_action[np.argmax(self.policy[list_action])]
        if len(temp)<2:
            temp = np.zeros(AMOUNT_ACTION)
        temp[action]+=1
        if getReward(state)==-1:
            pass
        else:
            if getReward(state)==1:
                self.count+=temp
                self.reward+=temp
                self.policy = self.reward / self.count
            elif getReward(state)==0:
                self.count+=temp
                self.policy = self.reward / self.count
        return action,temp,per
    def play(self,state,temp,per):
        list_action = getValidActions(state)
        list_action = np.where(list_action == 1)[0]
        action = list_action[np.argmax(self.policy[list_action])]
        return action,temp,per
    def train(self):
        list_player = [self.run] + [self.agent_random]*(getAgentSize()-1)
        normal_main(list_player,self.num_games,[0])
    
pseudo_best = AgentPolicy(2000)
pseudo_best.train()
class GeneticAlgorithm:
    def __init__(self):
        self.best = pseudo_best
        self.list_best_agent = []
        self.list_agent = []
        self.fitness_score = []
        self.top_fitness_score = []
    class Agent:
        """Agent with random weights"""
        def __init__(self):
            self.num_neurons = 64
            r1 = np.sqrt(2) * np.sqrt(6/(LEN_STATE + self.num_neurons))
            r2 = np.sqrt(2) * np.sqrt(6/(self.num_neurons+AMOUNT_ACTION))
            if np.random.rand()>=0.5:
                self.weights1 = np.random.uniform(low=-r1,high=r1,size= (LEN_STATE,self.num_neurons))
                self.weights2 = np.random.uniform(low=-r2,high=r2,size= (self.num_neurons,AMOUNT_ACTION))
            else:
                self.weights1 = np.random.randn(LEN_STATE,self.num_neurons)* r1 / np.sqrt(3)
                self.weights2 = np.random.randn(self.num_neurons,AMOUNT_ACTION) * r2 / np.sqrt(3)
            self.id = np.random.rand()
        def play(self,state,temp,per):
            list_action = getValidActions(state)
            list_action = np.where(list_action == 1)[0]
            hidden1 = state.dot(self.weights1)
            hidden2 = np.maximum(hidden1,0)
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
    
    def Mutation(self,Agent1,rate=0.01):
        """Small chance of changing some weights of child Agent"""
        Dummy = self.Agent()
        choice1 = np.random.choice([0,1],size=Dummy.weights1.shape,p=[rate,1-rate]).astype(bool)
        choice2 = np.random.choice([0,1],size=Dummy.weights2.shape,p=[rate,1-rate]).astype(bool)
        Agent1.weights1,Agent1.weights2 =  np.where(choice1,Agent1.weights1,Dummy.weights1),np.where(choice2,Agent1.weights2,Dummy.weights2)
        return Agent1
    def ChildAgent(self,*Agent,mutation_rate = 1):
        """Create Agent using Crossover and Mutation"""
        Agent1,Agent2 = np.random.choice(Agent,size=2,replace=False)
        Child = self.UniformCrossover(Agent1,Agent2)
        if np.random.rand() > mutation_rate:
            return Child
        else:
            return self.Mutation(Child)
    
    def agent_random(self,state,temp,per):
        """Agent random."""
        list_action = getValidActions(state)
        list_action = np.where(list_action == 1)[0]
        idx = random.randrange(len(list_action))
        return list_action[idx],temp,per
    
    def calculateFitness(self,Agent1: Agent,num_games=400):
        try:
            if Agent1.id==self.best.id:
                Agent1.fitness = 1
            else:
                lst = [Agent1.play,self.best.play]+[self.agent_random]*(getAgentSize()-2)
                sum = 0
                for _ in range(4):
                    rank = normal_main(lst,num_games,[0])[0]
                    sum+=rank[0] / rank[1]
                Agent1.fitness = sum / 4
        except:
            lst = [Agent1.play,self.best.play]+[self.agent_random]*(getAgentSize()-2)
            sum = 0
            for _ in range(4):
                rank = normal_main(lst,num_games,[0])[0]
                sum+=rank[0] / rank[1]
            Agent1.fitness = sum / 4

        return Agent1.fitness
    
    def chooseTopAgent(self,list_agent_)->list:
        """Return list top Agent and their fitness score"""
        self.fitness_score = np.array([self.calculateFitness(agent,num_games=400) for agent in list_agent_])
        return np.array(list_agent_)[np.argsort(self.fitness_score)][-5:],(self.fitness_score[np.argsort(self.fitness_score)][-5:])

    def nextGeneration(self,random=True):
        if random==False:
            for agent in self.list_best_agent:
                self.list_agent.append(agent)
            while len(self.list_agent)!=self.num_agents:
                Agent1,Agent2 = np.random.choice(self.list_best_agent,p=softmax(self.top_fitness_score),size=2,replace=False)
                self.list_agent.append(self.ChildAgent(Agent1,Agent2))
            for _ in range(3):
                self.list_agent.append(self.Agent())
        elif random==True:
            for agent in self.list_best_agent:
                self.list_agent.append(agent)
            for _ in range(3):
                self.list_agent.append(self.Agent())
            while len(self.list_agent)!=self.num_agents:
                Agent1,Agent2 = np.random.choice(self.list_best_agent,size=2,replace=False)
                self.list_agent.append(self.ChildAgent(Agent1,Agent2))
    def Gen0(self):
        while len(self.list_agent)!=self.num_agents:
            self.list_agent.append(self.Agent())
    def reset(self):
        self.list_agent = []
        self.fitness_score = []
    def train(self,num_gen = 5,num_agents=20):
        self.num_agents = num_agents
        for num in range(num_gen):
            if num==0:
                self.Gen0()
                self.list_best_agent,self.top_fitness_score = self.chooseTopAgent(self.list_agent)
                self.best = self.list_best_agent[-1]
                np.savez(os.path.join(path_save_player,'Best.npz'),w1=self.best.weights1,w2=self.best.weights2)
                self.reset()
            else:
                if np.random.rand()>0.2:
                    self.nextGeneration(random=False)
                else:
                    self.nextGeneration(random=True)
                self.list_best_agent,self.top_fitness_score = self.chooseTopAgent(self.list_agent)
                best_temp = self.list_best_agent[-1]
                if self.best == best_temp:
                    pass
                else:
                    fitness = self.calculateFitness(best_temp,num_games=3000)
                    if fitness > 1:
                        self.best = best_temp
                    else:
                        pass
                np.savez(os.path.join(path_save_player,'Best.npz'),w1=self.best.weights1,w2=self.best.weights2)
                self.reset()

            try:
                with open(os.path.join(path_save_player,'num_gen.json'),'r') as f:
                    data = json.load(f)
                    data+=1
                with open(os.path.join(path_save_player,'num_gen.json'),'w') as f:
                    json.dump(data,f)
            except:
                num = 1
                with open(os.path.join(path_save_player,'num_gen.json'),'w') as f:
                    json.dump(num,f)

model = GeneticAlgorithm()
def train(num_gen=5000,num_agents=20):
    model.train(num_gen=num_gen,num_agents=num_agents)
def test(state,temp,per):
    if len(temp)<2:
        player = 'Dat_270922'
        path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
        data = np.load(os.path.join(path_save_player,'Best.npz'))
        temp = [data['w1'],data['w2']]

    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    hidden1 = state.dot(temp[0])
    hidden2 = np.maximum(hidden1,0)
    values = hidden2.dot(temp[1])
    action = list_action[np.argmax(values.reshape(-1)[list_action])]
    return action,temp,per