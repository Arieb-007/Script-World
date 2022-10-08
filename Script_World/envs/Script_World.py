import pandas as pd
import json
import os
import gym
import networkx as nx
import numpy as np
import random

from gym import spaces
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#plt.style.use("dark_background")
#print("loading sbert model...")
#sbert_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
# sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")
#print("sbert model loaded...")
class ScriptWorldEnv(gym.Env) :

      def __init__(self,scn,no_of_actions,allowed_wrong_actions,hop,seed,disclose_state_node):  #e.g scn = bake a cake
        self.scn = scn 
        self.scn_graph = self.create_graph(scn)
        self.disclose_state_node=disclose_state_node
        self.cg = self.create_compact_graph(self.scn_graph)
        self.state_node = self.scn_graph['nodes'][0]['id']
        self.state = self.state_node.partition("_")[0]
        #print("The State is :")
        #print(self.state)
        self.t = 0
        g = self.cg
        self.cg.nodes[self.state]['no'] = 0
        self.dfs_init(g,self.state,1)
        self.dfs_max(g,self.state,1)
        self.Victory_node=self.cg.nodes['Victory']['no']
        self.wc = allowed_wrong_actions
        self.w_count = 0
        self.per_clp = 0
        self.seq_no = 0
        self.hop = hop
        self.seed  = seed
        self.done = False
        self.quest = self.scn_graph['nodes'][0]['quest']
        self.no_of_actions = no_of_actions
        self.action_spaces = []
        if disclose_state_node :
          self.observation_space=spaces.Box(low=-10000,high=10000,shape=((no_of_actions+1)*384,))
          self.action_space=spaces.Discrete(no_of_actions)
        else :
          self.observation_space=spaces.Box(low=-10000,high=10000,shape=((no_of_actions)*384,))
          self.action_space=spaces.Discrete(no_of_actions)
        self.num_envs=6
        A = []
        for d in self.scn_graph['links']:
          
          if(d['source']==self.state_node):
                A.append(d['target'])


        np.random.seed(seed)
        self.state_node = random.choice(A)
        for node in self.scn_graph['nodes']:
          
          if(node['id']==self.state_node):
                self.action_spaces.append(node['action'])
                
                break
        
        count = 0
        #negtive sampling of actions
        while(count < self.no_of_actions-1):
          node = random.choice(self.scn_graph['nodes'])
          if(abs(self.cg.nodes[node['id'].partition("_")[0]]['no']-self.cg.nodes[self.state]['no'])<=4) : continue
          if(node['id']=="Victory"): continue
          if(node['id'].partition("_")[0]!=self.state_node.partition("_")[0] and node['type']=='slot') :
            if(node['action'] in self.action_spaces) :continue
            count+=1           
            self.action_spaces.append(node['action'])
        random.shuffle(self.action_spaces)
      
      def preprocess_state(self, state):
        #print(state)
        if len(state)==0:
          if self.disclose_state_node :
            return [0]*(self.no_of_actions+1)*384
          else :
            return [0]*(self.no_of_actions)*384
        embeddings = sbert_model.encode(state)
        #print(embeddings.shape)
        return np.concatenate(embeddings, axis=0)

      def preprocess_states(self, states):
          embeddings = []
          for state in states:
              #print(state)
              embeddings.append(self.preprocess_state(state=state))
          return embeddings
      def step(self , action) :

        self.done = False
        #print(action)
        for node in self.scn_graph['nodes']:
          
          if(node['id']==self.state_node):
                self.right = node['action']
                
                break

        if(self.action_spaces[action]==self.right) :
            self.reward = 0
            self.w_count = 0
            for edge in self.scn_graph['links']:
              if(edge['source']==self.state_node) :
                 self.state_node = edge['target']
                 break

            if(self.state_node[-1]=='l' and self.state_node[-2]=='_') :  #it is a leave node,need more transition ,random seed will be used
                
                p_t = []
                for edge in self.scn_graph['links']:
                  if(edge['source']==self.state_node) :
                      p_t.append(edge['target'])
                np.random.seed(self.seed)
                self.state_node = random.choice(p_t)
                
                #on entry node now 
                if(self.state_node!='Victory'):
                     #percentage following
                     
                     
                     p_t = []
                     for edge in self.scn_graph['links']:

                       if(edge['source']==self.state_node) :
                          p_t.append(edge['target'])

                     np.random.seed(self.seed)
                     self.state_node = random.choice(p_t)

                        
        else :
              self.w_count += 1
              self.reward = -1
              n_cg = self.state_node.partition('_')[0]
              no = self.cg.nodes[n_cg]['no']
              no = max(0,no-self.hop)
              if(self.hop!=-1):
                SN = list(self.cg.nodes)
                random.shuffle(SN)
                for node in SN :
                  try:
                    if(self.cg.nodes[node]['no']==no):
                       self.state_node = node+'_'+'e'
                       break

                  except:
                       continue
                #now on entry node
                
                p_t = []
                for edge in self.scn_graph['links']:

                  if(edge['source']==self.state_node) :

                       p_t.append(edge['target'])

                np.random.seed(self.seed*3)
                self.state_node = random.choice(p_t)
              

        self.state = self.state_node.partition("_")[0]

        self.per_clp = self.cg.nodes[self.state_node.partition("_")[0]]['no']/self.cg.nodes['Victory']['no'] 
        if(self.state_node=='Victory' or self.w_count==self.wc) :   #Terminal state
             
              self.done = True
              self.per_clp = 1
              if(self.w_count==self.wc) : self.reward += -5
              else : self.reward += 10
        #print(self.cg.nodes[self.state]['no'])    
        self.completion_percentage = (self.cg.nodes[self.state]['no'] / self.Victory_node) * 100
        #print(self.completion_percentage)
        #new action space

        self.action_spaces = []
        if(not self.done) : 
          A = []
          for node in self.scn_graph['nodes']:
            
            if(node['id']==self.state_node and node['type']=='slot'):
                 self.action_spaces.append(node['action'])
                 break
          #print(self.state_node)
          np.random.seed(self.seed)
          count = 0
          while(count < self.no_of_actions-1):
            node = random.choice(self.scn_graph['nodes'])
            if(abs(self.cg.nodes[node['id'].partition("_")[0]]['no']-self.cg.nodes[self.state]['no'])<=4) : continue
            if(node['id']=="Victory"): continue
            if(node['id'].partition("_")[0]!=self.state_node.partition("_")[0] and node['type']=='slot') :
              if(node['action'] in self.action_spaces) :continue
              count+=1           
              self.action_spaces.append(node['action'])
      
        random.shuffle(self.action_spaces)
       

        #return  self.action_space , self.reward , self.done , self.quest
        #print(self.state_node)
        if self.disclose_state_node:

            return (
                self.preprocess_state([self.state_node] + self.action_spaces),
                self.reward,
                self.done,
                {}#self.quest,
            )
        else:
            return (
                self.preprocess_state(self.action_spaces),
                self.reward,
                self.done,
                {}#self.quest,
            )
      
      def reset(self) :

        
        self.state_node = self.scn_graph['nodes'][0]['id']
        self.w_count = 0
        self.done = False
        self.action_spaces = []
        self.per_clp = 0
        A = []
        for d in self.scn_graph['links']:
          
          if(d['source']==self.state_node):
                A.append(d['target'])


        np.random.seed(self.seed)
        self.state_node = random.choice(A)
        for node in self.scn_graph['nodes']:
          
          if(node['id']==self.state_node):
                #print(node)
                self.action_spaces.append(node['action'])
                break
        
        count = 0
        while(count < self.no_of_actions-1):
          node = random.choice(self.scn_graph['nodes'])
          if(abs(self.cg.nodes[node['id'].partition("_")[0]]['no']-self.cg.nodes[self.state]['no'])<=4) : continue
          if(node['id']=="Victory"): continue
          if(node['id']!=self.state_node and node['type']=='slot') :
            if(node['action'] in self.action_spaces) :continue
            count+=1           
            self.action_spaces.append(node['action'])

        self.state = self.state_node.partition("_")[0]
        
        random.shuffle(self.action_spaces)
        
        #return self.action_space
        #print(self.action_spaces)
        if self.disclose_state_node:
            return self.preprocess_state( [self.state_node] + self.action_spaces)
        else:
            return self.preprocess_state( self.action_spaces)
          

      def create_graph(self,scn):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = scn + '.new.xml_.json'
        with open(os.path.join(dir_path,'json',file), 'r') as myfile:
           data=myfile.read()

        json_str = json.loads(data)
        df = json.loads(json_str)
  

        return df
      def create_compact_graph(self,D):         
        g = D
        G = nx.DiGraph()
        for n in g['nodes']:    
            #leave node
         if(n['id']=='Victory'):
            G.add_node(n['id'])
            continue

         if(n['id'][-1]=='l' and n['id'][-2]=='_'):
        
           G.add_node(n['id'][:-2])
           for d in g['links']:
          
             if(d['source']==n['id']):
           
                if(d['target']=='Victory'):
                   G.add_node(d['target'])
                   G.add_edge(n['id'][:-2],d['target'])
                   continue
                G.add_node(d['target'][:-2])
                G.add_edge(n['id'][:-2],d['target'][:-2]) 

         if(n['id'][-1]=='e' and n['id'][-2]=='_'):
             count=0
             for d in g['links']:
                if(d['source']==n['id']):
                       G.add_node(n['id'][:-2])
                       count+=1
                       G.nodes[n['id'][:-2]]['split_ways'] = count
        
        for i in G.nodes :
          G.nodes[i]['no'] = 0
        return G      

      def dfs_init(self,g,n,t):
       
        if(n=='Victory'):      
               self.cg.nodes[n]['no'] = t
                   
               return

        else :
      
          for i in self.cg[n].keys() :
             self.cg.nodes[i]['no'] = t
             self.dfs_init(g,i,t+1)
               

      def dfs_max(self,g,n,t):
         
         if(n=='Victory'):      
             self.cg.nodes[n]['no'] = max(t,self.cg.nodes[n]['no'])
             
          
             return 

         else :
            for i in self.cg[n].keys() :
                 self.cg.nodes[i]['no'] = max(t,self.cg.nodes[i]['no'])
                 self.dfs_max(g,i,t+1)
