from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
					
class QLearningAgent(ReinforcementAgent):
	"""
		Q-Learning Agent
		
		Functions you should fill in:
			- getQValue
			- getAction
			- getValue
			- getPolicy
			- update
			
		Instance variables you have access to
			- self.epsilon (exploration prob)
			- self.alpha (learning rate)
			- self.gamma (discount rate)
		
		Functions you should use
			- self.getLegalActions(state) 
				which returns legal actions
				for a state
	"""
	def __init__(self, **args):
		"You can initialize Q-values here..."
		ReinforcementAgent.__init__(self, **args)
		self.qvals = {}
		
	
	def getQValue(self, state, action):
		"""
			Returns Q(state,action)    
			Should return 0.0 if we never seen
			a state or (state,action) tuple 
		"""
		return self.qvals.get(state, Counter())[action]
	
		
	def getValue(self, state):
		"""
			Returns max_action Q(state,action)        
			where the max is over legal actions.  Note that if
			there are no legal actions, which is the case at the
			terminal state, you should return a value of 0.0.
		"""
		values = self.qvals.get(state, Counter()).values()
		if values:
			return max(values)
		return 0.0

	def getPolicy(self, state):
		"""
			Compute the best action to take in a state.  Note that if there
			are no legal actions, which is the case at the terminal state,
			you should return None.
		"""
		legalActions = self.getLegalActions(state)

		if not legalActions:
			return None

		if state not in self.qvals:
			return random.choice(legalActions)

		sortedActions = self.qvals[state].sortedKeys()

		if sortedActions:
			return sortedActions[0]

		return None
		
	def getAction(self, state):
		"""
			Compute the action to take in the current state.  With
			probability self.epsilon, we should take a random action and
			take the best policy action otherwise.  Note that if there are
			no legal actions, which is the case at the terminal state, you
			should choose None as the action.
		"""  
		# Pick Action
		legalActions = self.getLegalActions(state)
		
		if legalActions and util.flipCoin(self.epsilon):
			return random.choice(legalActions)

		else:
			return self.getPolicy(state)
		
		return None
	
	def update(self, state, action, nextState, reward):
		"""
			The parent class calls this to observe a 
			state = action => nextState and reward transition.
			You should do your Q-Value update here
			
			NOTE: You should never call this function,
			it will be called on your behalf
		"""
		
		if not state in self.qvals:
			self.qvals[state] = Counter()
		if not nextState in self.qvals:
			self.qvals[nextState] = Counter()

		old_qval = self.qvals[state][action]

		new_qval = (1 - self.alpha) * old_qval
		new_qval += self.alpha * self.gamma * self.getValue(nextState)

		
class PacmanQAgent(QLearningAgent):
	"Exactly the same as QLearningAgent, but with different default parameters"
	
	def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
		"""
		These default parameters can be changed from the pacman.py command line.
		For example, to change the exploration rate, try:
				python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
		
		alpha    - learning rate
		epsilon  - exploration rate
		gamma    - discount factor
		numTraining - number of training episodes, i.e. no learning after these many episodes
		"""
		args['epsilon'] = epsilon
		args['gamma'] = gamma
		args['alpha'] = alpha
		args['numTraining'] = numTraining
		QLearningAgent.__init__(self, **args)

	def getAction(self, state):
		"""
		Simply calls the getAction method of QLearningAgent and then
		informs parent of action for Pacman.  Do not change or remove this
		method.
		"""
		action = QLearningAgent.getAction(self,state)
		self.doAction(state,action)
		return action

		
class ApproximateQAgent(PacmanQAgent):
	"""
		 ApproximateQLearningAgent
		 
		 You should only have to overwrite getQValue
		 and update.  All other QLearningAgent functions
		 should work as is.
	"""
	def __init__(self, extractor='IdentityExtractor', **args):
		self.featExtractor = util.lookup(extractor, globals())()
		PacmanQAgent.__init__(self, **args)

		# You might want to initialize weights here.
		"*** YOUR CODE HERE ***"
		
	def getQValue(self, state, action):
		"""
			Should return Q(state,action) = w * featureVector
			where * is the dotProduct operator
		"""
		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()
		
	def update(self, state, action, nextState, reward):
		"""
			 Should update your weights based on transition  
		"""
		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()
		
	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		PacmanQAgent.final(self, state)
		
		# did we finish training?
		if self.episodesSoFar == self.numTraining:
			# you might want to print your weights here for debugging
			"*** YOUR CODE HERE ***"
			pass
