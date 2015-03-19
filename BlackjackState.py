from BlackjackGlobals import *
import PointFunctions as pf

class BlackjackState:
	
	minGroupDist = max(imageX,imageY)/4

	"""
	Class that determines the state of a game of blackjack given the
	list of cards and card group assignments
	"""
	def __init__(self, cards, cardGroups):
		self.dealerPos = None
		self.isValid = self._isStateValid(cards, cardGroups)
		if not self.groups is None:
			print self.groups
			self._computeDealersAndPlayers()
		else:
			self.dealers = []
			self.players = []

	def __eq__(self, other):
		if other==None:
			return False
		return str(self.groups)==str(other.groups)

	"""
	Filters card groups that are the dealer and are not the dealer
	"""
	def _computeDealersAndPlayers(self):
		self.dealers = filter(lambda g:g.isDealer, self.groups)
		self.players = filter(lambda g:not g.isDealer, self.groups)

	"""
	A state is invalid under the conditions stated in the comments below
	"""
	def _isStateValid(self, cards, cardGroups):
		self.groups = None

		# state not valid if no cards
		if len(cards)==0:
			return False

		# group cards by given card group indices
		groups = self._groupCards(cards, cardGroups)

		# state not valid if dealer or player doesn't have cards
		hasDealer = self._identifyDealer(groups)
		if len(groups) < 2:
			return False

		# state not valid if any player group has <2 cards
		if not all(map(lambda g:g.isValid, groups)):
			return False

		# If any player produces an invalid move
		if hasDealer:
			for group in groups:
				if not group.isDealer:
					if group.computeMove(self.dealer)=="I":
						print "FOUND INVALID STATE", self.group, self.dealer
						return False

		# State is valid
		self.groups = groups
		return True


	"""
	Group list of BlackjackCard objects by their given group indices
	"""
	def _groupCards(self, cards, cardGroups):
		groups = {g:[] for g in set(cardGroups)}
		for i,c in enumerate(cards):
			groups[cardGroups[i]].append(cards[i])
		return map(lambda g:CardGroup(g), groups.values())

	"""
	Identify which card group, if any is the dealer
	"""
	def _identifyDealer(self, groups):
		counts = map(lambda g:len(g.cards), groups)
		if counts.count(1) == 1:
			self.dealerPos = counts.index(1)
		# identify dealer by distance
		elif not self.dealerPos is None:
			dealerDists = map(lambda g:pf.dist(self.dealerPos, g.center), groups)
			if min(dealerDists)>BlackjackState.minGroupDist:
				return False
			self.dealerPos = groups[dealerDists.index(min(dealerDists))]
		else:
			return False
		self.dealer = groups[self.dealerPos]
		self.dealer.setDealer()
		self.dealerPos = self.dealer.center
		return True

	"""
	Sets the group index of the dealer
	"""
	def setDealer(self, group):
		if group in self.groups:
			self.groups[self.groups.index(group)].isDealer = True
			self._computeDealersAndPlayers()

	"""
	Determines if the given next state is a valid future state
	from this state after the player has standed
	"""
	def isNextState(self, next):
		# only consider valid states
		if not self.isValid or not next.isValid:
			return False, None
		# currently only consider when there's two groups
		if len(next.groups)!=2 or len(self.groups)!=2:
			return False, None
		# next state should not be able to recognise a dealer
		if len(next.dealers)!=0:
			return False, None
		# if two states are exactly the same, it is not a future state
		if self == next:
			return False, None

		# only matches if player group matches and dealer has at least same cards 
		playerMatches = False
		dealerMatches = False
		dealerGroup = None
		for g in next.groups:
			if g == self.players[0]:
				playerMatches = True
			else:
				dealerMatches = True
				for cardStr in self.dealers[0].cardStrings:
					if cardStr not in g.cardStrings:
						dealerMatches = False
						break
				if dealerMatches:
					dealerGroup = g

		return (playerMatches and dealerMatches, dealerGroup)

class CardGroup:
	"""
	Class which analyzes a list of cards and extracts useful information,
	such as the score and the blackjack basic strategy move
	"""
	def __init__(self, cards):
		self.cards = cards
		self.cardStrings = map(lambda c:c.name, cards)
		self.cardStrings.sort()
		self.center = tuple(pf.avg(map(lambda c:c.center, cards)))
		self.isDealer = False
		self.isValid = len(cards)>1
		self._computeScore()
		self.move = "I"
		self.isBlackjack = self.scoreKey=="21S" and len(self.cards)==2

	def __repr__(self):
		g = "D" if self.isDealer else "P"
		g += "("+",".join(self.values)+")"
		g += "="+self.scoreKey
		if not self.isDealer:
			g += ":"+self.move
		return g

	def __eq__(self, other):
		#return set(self.cards) == set(other.cards) and self.isDealer==other.isDealer 
		return self.cardStrings == other.cardStrings and self.isDealer==other.isDealer 

	"""
	Computes a numerical and a string score for the hand
	"""
	def _computeScore(self):
		values = map(lambda c:c.name[0], self.cards)
		score = sum(map(lambda v:cardValueMap[v], values))
		# make As soft when over 21
		self.isDouble = len(values)==2 and values[0]==values[1]
		if score>21 and "A" in values:
			values[values.index("A")] = "Ahard"
			score -= 10
		self.isSoft = "A" in values
		self.values, self.score = values, score
		self.isBust = self.score>=21
		self.isBlackjack = self.score==21
		# key used to lookup basic strategy dictionary
		self.scoreKey = str(self.score)
		"""
		Don't consider doubles for now
		Double Down and Split are not yet implemented in BlackjackSpeaker
		"""
		"""
		if self.isDouble:
			self.scoreKey += "D"
		"""
		if self.isSoft:
			self.scoreKey += "S"

	"""
	Marks this group as a dealer
	"""
	def setDealer(self):
		self.isDealer = True
		self.isValid = True
		if len(self.values)==1:
			self.scoreKey = str(cardValueMap[self.values[0]])

	"""
	Computes move based on basic strategy, given the dealer's hand
	"""
	def computeMove(self, dealer):
		if len(dealer.cards)>1:
			self.move = "W"# wait
		elif self.score>21:
			self.move = "B"# bust
		else:
			key = (self.scoreKey, dealer.scoreKey)
			if key in basicStrategy:
				self.move = basicStrategy[key]
			else:
				self.move = "I"# invalid state
