# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 23:13:58 2016

@author: Kai
"""

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class QTable(object):
    def __init__(self):
        self.Q = dict()
        
    def get(self, state, action):
        key = (state, action)
        return self.Q.get(key, None)

    def set(self, state, action, q):
        key = (state, action)
        self.Q[key] = q

    def report(self):
        for k, v in self.Q.items():
            print k, v

class QAlgorithm(Agent):
    def __init__(self, epsilon = 0.1, alpha = 0.5, gamma = 0.6):
        self.Q = QTable()
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.valid_actions = Environment.valid_actions
        
    def q_move(self, state):
        if random.random() < self.eps: 
            action = random.choice(self.valid_actions)
        else:
            q = [self.Q.get(state, acts) for acts in self.valid_actions]
            max_q = max(q)
            
            if q.count(max_q) > 1:
                best_actions = [i for i in range(len(self.valid_actions)) if q[i] == max_q]
                act_index = random.choice(best_actions)
            else:
                act_index = q.index(max_q)
            action = self.valid_actions[act_index]
        return action
        
    def q_learn(self, state, action, reward, new_q):
        q = self.Q.get(state, action)
        if q is None:
            q = reward
        else:
            q = q + self.alpha* new_q
    
        self.Q.set(state, action, q)
    
    def q_future_reward(self, state, action, next_state, reward):
        q = [self.Q.get(next_state, a) for a in self.valid_actions]
        future_rewards = max(q)
        if future_rewards is None:
            future_rewards = 0.0
            
        self.q_learn(state, action, reward, reward - self.gamma * future_rewards)
        self.Q.report()
        
class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.total_reward = 0
        self.next_waypoint = None
        self.QLearner = QAlgorithm(epsilon=0.03, alpha = 0.1, gamma = 0.9)
        self.next_state = None
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_action = None
        self.next_state = None
        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        #env_states = self.env.agent_states[self]

        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        action = self.QLearner.q_move(self.state)        
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        
        new_inputs = self.env.sense(self)
        #env_states = self.env.agent_states[self]

        self.next_state =  (new_inputs['light'], new_inputs['oncoming'], self.next_waypoint)
        # TODO: Learn policy based on state, action, reward

        self.QLearner.q_future_reward(self.state, action, self.next_state, reward)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
