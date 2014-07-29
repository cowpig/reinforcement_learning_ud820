import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
	"""
			* Please read learningAgents.py before reading this.*

			A ValueIterationAgent takes a Markov decision process
			(see mdp.py) on initialization and runs value iteration
			for a given number of iterations using the supplied
			discount factor.
	"""
	def __init__(self, mdp, discount = 0.9, iterations = 100):
		"""
			Your value iteration agent should take an mdp on
			construction, run the indicated number of iterations
			and then act according to the resulting policy.
		
			Some useful mdp methods you will use:
					mdp.getStates()
					mdp.getPossibleActions(state)
					mdp.getTransitionStatesAndProbs(state, action)
					mdp.getReward(state, action, nextState)
		"""
		self.mdp = mdp
		self.discount = discount
		self.iterations = iterations
		self.values = util.Counter() # A Counter is a dict with default 0

		for state in self.mdp.getStates():
			if mdp.isTerminal(state):
				self.values[state] = self.mdp.getReward(state, None, state)
		 
		for i in xrange(iterations):
			print "\n\nITERATION {}\n".format(i)
			updates = {}
			for state in self.mdp.getStates():
				print "--state {}--".format(state)
				potential_updates = []
				for action in self.mdp.getPossibleActions(state):
					if i == 1 and state == (2,2) and action == ("east"):
						import pdb; pdb.set_trace()
					potential_updates.append(self.getQValue(state, action))
					print "q-val for action {} is {}".format(action, potential_updates[-1])
					
				if potential_updates:
					print "potential_updates:"
					print potential_updates
					updates[state] = max(potential_updates)

			print "updates:"
			print updates
			self.values.update(updates)
			print "new values:"
			print self.values
		
	def getValue(self, state):
		"""
			Return the value of the state (computed in __init__).
		"""
		return self.values[state]


	def getQValue(self, state, action):
		"""
			The q-value of the state action pair
			(after the indicated number of value iteration
			passes).  Note that value iteration does not
			necessarily create this quantity and you may have
			to derive it on the fly.
		"""
		q = self.values[state]
		print "\tq-value calc for {}, {}:".format(state, action)
		for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
			# reward = self.mdp.getReward(state, action, next_state)
			reward = self.values[next_state]
			print "\t\treward from state {} = {}".format(next_state, reward)
			q += self.discount * prob * reward

		return q


	def getPolicy(self, state):
		"""
			The policy is the best action in the given state
			according to the values computed by value iteration.
			You may break ties any way you see fit.  Note that if
			there are no legal actions, which is the case at the
			terminal state, you should return None.
		"""
		best_action = None
		best_value = None
		for action in self.mdp.getPossibleActions(state):
			value = self.getQValue(state, action)
			if best_action == None or best_value < value:
				best_action = action
				best_value = value

		return best_action


	def getAction(self, state):
		"Returns the policy at the state (no exploration)."
		return self.getPolicy(state)
	
