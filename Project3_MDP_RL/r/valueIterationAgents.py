# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # MDP Iteration
        for i in range(self.iterations):  # number of iterations
            tempvalues = util.Counter()  # Temporary storage for values
            for state in self.mdp.getStates():  # Iterate through all states
                if not self.mdp.getPossibleActions(state):  # If the state is terminal, skip it
                    tempvalues[state] = 0 # Terminal states have a value of 0
                    continue
                
                q_values = max(self.computeQValueFromValues(state, action)  # Compute Q-values for each action
                    for action in self.mdp.getPossibleActions(state)) # possible actions for the state)
                tempvalues[state] = q_values  # Store the maximum Q-value for the state
            
            self.values = tempvalues # new values
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        PossibleTransitions = self.mdp.getTransitionStatesAndProbs(state, action) # Get possible transitions for the state and action
        expected_values = [prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])for next_state, prob in PossibleTransitions] # Calculate expected values for each transition
        return sum(expected_values)  # Sum the expected values to get the Q-value
    
        # Sum the expected values to get the Q-value
        return sum(expected_values)

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.
        You may break ties any way you see fit. Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        # If no actions are possible, return None
        if not self.mdp.getPossibleActions(state):
            return None
        # Use max with a key function to find the best action
        return max(self.mdp.getPossibleActions(state), key=lambda action: self.computeQValueFromValues(state, action)) #return the action that maximizes the Q-value for the state
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Priority queue for states to update
        pq = util.PriorityQueue()
        
        # Predecessors of each state
        predecessors = {state: set() for state in self.mdp.getStates()}
        
        #TODO: map each state to its predecessors
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        # Adding next_state to the predecessors of state
                        predecessors[next_state].add(state)
        
        #TODO: The initialization of the priority queue with the absolute difference between the current value and the new value
        for state in self.mdp.getStates():
            # If the state is terminal, skip it
            if not self.mdp.getPossibleActions(state):
                continue
            
            # Compute Q-values for each action
            q_values = max(
                self.computeQValueFromValues(state, action) # Compute Q-values for each action
                for action in self.mdp.getPossibleActions(state) # possible actions for the state
            )
            
            # Calculate the difference between the current value and the new value
            diff = abs(self.values[state] - q_values)
            
            pq.update(state, -diff)
        
        # Number of iterations
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            
            # Get the state with the highest priority
            curr_state = pq.pop()
            
            if not self.mdp.getPossibleActions(curr_state):
                continue
            
            # Update the value of the current state
            q_values = max(
                self.computeQValueFromValues(curr_state, action) # Compute Q-values for each action
                for action in self.mdp.getPossibleActions(curr_state)
            )
            self.values[curr_state] = q_values
            
            # For each predecessor of the current state
            for pred in predecessors[curr_state]:
                if not self.mdp.getPossibleActions(pred):
                    continue
                
                # Calculate the difference between the current value and the new value
                q_values = max(
                    self.computeQValueFromValues(pred, action) # Compute Q-values for each action
                    for action in self.mdp.getPossibleActions(pred)
                )
                diff = abs(self.values[pred] - q_values) #difference btw value
                
                if diff > self.theta:
                    pq.update(pred, -diff) # Update the priority queue with the new diff