from BlackjackGlobals import *
import PointFunctions as pf

class CardGroup:

	def __init__(self, cards):
		self.cards = cards
		self.cardStrings = map(lambda c:c.name, cards)
		self.cardStrings.sort()
		self.center = tuple(pf.avg(map(lambda c:c.center, cards)))
		self.isDealer = False
		self.isValid = len(cards)>1
		self._computeScore()
		self.move = "I"
		self.isBlackjack = self.scoreKey=="21S"

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
		# Don't consider doubles for now
		"""
		if self.isDouble:
			self.scoreKey += "D"
		"""
		if self.isSoft:
			self.scoreKey += "S"

	def setDealer(self):
		self.isDealer = True
		self.isValid = True
		if len(self.values)==1:
			self.scoreKey = str(cardValueMap[self.values[0]])

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
				self.move = "I"# invalid, should not happen

class BlackjackState:
	
	minGroupDist = max(imageX,imageY)/4


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

	def _computeDealersAndPlayers(self):
		self.dealers = filter(lambda g:g.isDealer, self.groups)
		self.players = filter(lambda g:not g.isDealer, self.groups)

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

		if hasDealer:
			for group in groups:
				if not group.isDealer:
					if group.computeMove(self.dealer)=="I":
						print "FOUND INVALID STATE", self.group, self.dealer
						return False

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

	def setDealer(self, group):
		if group in self.groups:
			self.groups[self.groups.index(group)].isDealer = True
			self._computeDealersAndPlayers()
			print "SDJFKLSDJ", self.dealers
			if len(self.dealers) == 0:
				print "WHAT THE ACTUAL FUCKING FUCK"

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