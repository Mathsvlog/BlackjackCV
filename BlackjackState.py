from BlackjackGlobals import *
import PointFunctions as pf

class CardGroup:

	def __init__(self, cards):
		self.cards = cards
		self.center = tuple(pf.avg(map(lambda c:c.center, cards)))
		self.isDealer = False
		self.isValid = len(cards)>1
		self._computeScore()
		self.choice = "I"

	def __repr__(self):
		g = "D" if self.isDealer else "P"
		g += "("+",".join(self.values)+")"
		g += "="+self.scoreKey
		if not self.isDealer:
			g += ":"+self.choice
		return g

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
		if self.isDouble:
			self.scoreKey += "D"
		if self.isSoft:
			self.scoreKey += "S"

	def setDealer(self):
		self.isDealer = True
		self.isValid = True
		if len(self.values)==1:
			self.scoreKey = str(cardValueMap[self.values[0]])

	def computeChoice(self, dealer):
		if len(dealer.cards)>1:
			self.choice = "W"# wait
		elif self.score>21:
			self.choice = "B"# bust
		else:
			key = (self.scoreKey, dealer.scoreKey)
			if key in basicStrategy:
				self.choice = basicStrategy[key]
			else:
				self.choice = "I"# invalid, should not happen

class BlackjackState:
	
	minGroupDist = max(imageX,imageY)/4


	def __init__(self, cards, dealerPos=None):
		self.dealerPos = dealerPos
		self.isValid = self._isStateValid(cards)
		if not self.groups is None:
			print self.groups

	def _isStateValid(self, cards):
		self.groups = None

		# state not valid if no cards
		if len(cards)==0:
			return False

		# group cards by proximity
		groups = self._groupCards(cards)

		# state not valid if dealer or player doesn't have cards
		if len(groups) < 2 or not self._identifyDealer(groups):
			return False

		# state not valid if any player group has <2 cards
		if not all(map(lambda g:g.isValid, groups)):
			return False

		for group in groups:
			if not group.isDealer:
				if group.computeChoice(self.dealer)=="I":
					print "FOUND INVALID STATE", self.group, self.dealer
					return False

		self.groups = groups
		return True


	"""
	Group list of BlackjackCard objects by their proximity
	"""
	def _groupCards(self, cards):
		groupIndex = {i:i for i in range(len(cards))}
		# for each card
		for i,c1 in enumerate(cards):
			closestCard = i
			closestDist = BlackjackState.minGroupDist
			# find closest card
			for j,c2 in enumerate(cards):
				if i!=j:
					dist = pf.dist(c1.center, c2.center)
					if dist < closestDist:
						closestCard, closestDist = j, dist
			# combine the two groups
			if closestCard!=i:
				if groupIndex[i] > groupIndex[closestCard]:
					groupIndex[i] = groupIndex[closestCard]
				else:
					groupIndex[closestCard] = groupIndex[i]
		# create list of cards for each group
		groups = {g:[] for g in set(groupIndex.values())}
		for i,c in enumerate(cards):
			groups[groupIndex[i]].append(cards[i])
		return map(lambda g:CardGroup(g), groups.values())

	"""
	Identify which card group, if any is the dealer
	"""
	def _identifyDealer(self, groups):
		counts = map(lambda g:len(g.cards), groups)
		if counts.count(1) == 1:
			self.dealer = groups[counts.index(1)]
		# identify dealer by distance
		elif not self.dealerPos is None:
			dealerDists = map(lambda g:pf.dist(self.dealerPos, g.center), groups)
			if min(dealerDists)>BlackjackState.minGroupDist:
				return False
			self.dealer = groups[dealerDists.index(min(dealerDists))]
		else:
			return False
		self.dealer.setDealer()
		self.dealerPos = self.dealer.center
		return True
