"""
If you don't have pyttsx module installed, speark will print statements instead of speaking them
"""
try:
	import pyttsx
except:
	pass
from threading import Thread

class BlackjackSpeaker:

	repeatsRequired = 3
	missesRequired = 5
	"""
	Double Down and Split are not yet implemented in BlackjackSpeaker
	"""
	phrases = {"H": "Hit me.", "S":"Stand.", "P":"Split the cards.", "D":"Hit me.", "B":"I bust."}
	verbose = True

	def __init__(self):
		# pyttsx engine and properties
		try:
			self.engine = pyttsx.init()
			self.engine.setProperty("rate", 140)
			voices = self.engine.getProperty("voices")
			self.engine.setProperty("voice", voices[-1].id)
			self.hasSpeech = True
		except:
			self.hasSpeech = False

		self.lastState = None# current belief state of the speaker
		self.repeats = 0# if hits repeatsRequired, will speak move
		self.misses = 0# if hits missresRequired, will set lastState to current state
		self.thread = None# the separate thread handling speaking
		self.isWaiting = False
		self.phraseRemaining = []
		self.lastPhrase = ""

	"""
	Analyzes the current BlackjackState and tries to announce moves or the outcome of a game.
	"""
	def analyzeState(self, state):
		# When player is waiting on dealer
		if self.isWaiting:
			self._analyzeStateWaiting(state)

		# When player is trying to make a move
		else:
			# State matches the state from the last frame
			if state == self.lastState:
				self.repeats += 1
				self.misses = 0
				if self.repeats == BlackjackSpeaker.repeatsRequired and state.groups!=None:
					# figure out move
					player = filter(lambda group:not group.isDealer, state.groups)[0]
					if player.move in BlackjackSpeaker.phrases:
						phrase = ""
						if BlackjackSpeaker.verbose:
							phrase += self._verbosePhrase(state)
						phrase += BlackjackSpeaker.phrases[player.move]
						self.say(phrase)
						# Wait on dealer if standing
						if player.move == "S":
							self.isWaiting = True
							self.misses = 0
							self.repeats = 0
						self.lastState = state

			# State does not match the state from the last frame
			else:
				self.misses += 1
				if self.misses == BlackjackSpeaker.missesRequired:
					self.lastState = state
					self.repeats = 0
					self.misses = 0

		self.attemptPhrase()

	"""
	Player is waiting on dealer. Detect when dealer busts or reaches 17.
	Determine the outcome of the game and state the winner.
	"""
	def _analyzeStateWaiting(self, state):
		isNextState, dealerGroup = self.lastState.isNextState(state)
		if isNextState:
			self.repeats += 1
			self.misses = 0
			if self.repeats == BlackjackSpeaker.repeatsRequired:
				dScore = dealerGroup.score
				pScore = self.lastState.players[0].score
				if dScore < 17:
					if BlackjackSpeaker.verbose:
						self.say("Dealer currently has "+str(dScore)+". ", checkLast=True)
				else:
					phrase = ""
					if dScore > 21:
						if BlackjackSpeaker.verbose:
							phrase += "Dealer busts with "+str(dScore)+". "
						phrase += "I win."
					else:
						if BlackjackSpeaker.verbose:
							phrase += "Dealer stands with "+str(dScore)+". "
						if dScore > pScore:
							phrase += "I lose."
						elif dScore == pScore:
							if dScore == 21:
								if dealerGroup.isBlackjack == self.lastState.players[0].isBlackjack:
									phrase += "We tie."
								elif dealerGroup.isBlackjack:
									phrase += "I lose against blackjack."
								else:
									phrase += "I win with blackjack."
							else:
								phrase += "We tie."
						else:
							phrase += "I win."
					self.say(phrase)
					self.isWaiting = False
					self.lastState = None
				self.repeats = 0
		elif state == self.lastState:
			self.repeats = 0
			self.misses = 0
		else:
			if not state.isValid:
				self.misses += 1
			else:
				self.misses = 0
			if self.misses == BlackjackSpeaker.missesRequired*2:
				self.repeats = 0
				self.say("New round.")
				self.lastState = None
				self.isWaiting = False
				self.misses = 0

	"""
	Append the given phrase to the list of phrases to say
	"""
	def say(self, phrase, checkLast=False):
		if phrase not in self.phraseRemaining and (not checkLast or phrase != self.lastPhrase):
			self.phraseRemaining.append(phrase)

	"""
	Try to say the next phrase
	"""
	def attemptPhrase(self):
		if len(self.phraseRemaining) == 0:
			return
		if self.hasSpeech:
			self.engine.say(self.phraseRemaining[0])
			if self.thread == None or not self.thread.isAlive():
				self.thread = Thread(target=self.engine.runAndWait)
				self.thread.start()
				self.lastPhrase = self.phraseRemaining[0]
				self.phraseRemaining = self.phraseRemaining[1:]
		else:
			print self.phraseRemaining[0]
			self.phraseRemaining = self.phraseRemaining[1:]

	"""
	In verbose mode, announce the current state
	"""
	def _verbosePhrase(self, state):
		phrase = ""
		if len(state.dealers)!=1 or len(state.players)!=1:
			return phrase
		phrase += "I have "+str(state.players[0].score)
		phrase += " and dealer has "+str(state.dealers[0].score)+"; "
		return phrase
