import pyttsx
from threading import Thread


class BlackjackSpeaker:

	repeatsRequired = 5
	missesRequired = 5
	phrases = {"H": "Hit me.", "S":"Stand.", "P":"Split the cards.", "D":"Double down.", "B":"I bust."}
	verbose = True

	def __init__(self):
		# pyttsx engine and properties
		self.engine = pyttsx.init()
		self.engine.setProperty("rate", 140)
		voices = self.engine.getProperty("voices")
		self.engine.setProperty("voice", voices[-1].id)

		self.lastState = None# current belief state of the speaker
		self.repeats = 0# if hits repeatsRequired, will speak move
		self.misses = 0# if hits missresRequired, will set lastState to current state
		self.thread = None# the separate thread handling speaking
		self.isWaiting = False

	def analyzeState(self, state):
		# When player is waiting on dealer
		if self.isWaiting:
			isNextState, dealerGroup = self.lastState.isNextState(state)
			#print "NEXT_STATE", isNextState
			if isNextState:
				self.repeats += 1
				self.misses = 0
				if self.repeats >= BlackjackSpeaker.repeatsRequired:
					dScore = dealerGroup.score
					pScore = self.lastState.players[0].score
					if dScore < 17:
						if BlackjackSpeaker.verbose:
							self.say("Dealer currently has "+str(dScore)+". ")
					else:
						phrase = ""
						if dScore > 21:
							if BlackjackSpeaker.verbose:
								phrase += "Dealer busts with "+str(dScore)+". "
							phraw += "I win."
						else:
							if BlackjackSpeaker.verbose:
								phrase += "Dealer stands with "+str(dScore)+". "
							if dScore > pScore:
								phrase += "I lose."
							elif dScore == pScore:
								phrase += "We tie."
							else:
								phrase += "I win."
						self.say(phrase)
						self.isWaiting = False
						self.lastState = None
					self.repeats = 0
			elif state == self.lastState:
				self.repeats = 0
			else:
				#self.misses += 1
				self.repeats = 0
				if self.misses >= BlackjackSpeaker.missesRequired:
					self.isWaiting = False

		# When player is trying to make a move
		else:
			if state == self.lastState:
				self.repeats += 1
				self.misses = 0
				if self.repeats >= BlackjackSpeaker.repeatsRequired and state.groups!=None:
					# figure out move
					player = filter(lambda group:not group.isDealer, state.groups)[0]
					if player.move in BlackjackSpeaker.phrases:
						phrase = ""
						if BlackjackSpeaker.verbose:
							phrase += self._verbosePhrase(state)
						phrase += BlackjackSpeaker.phrases[player.move]
						if not self.say(phrase):
							self.repeats -= 1
						# Wait on dealer if standing
						if player.move == "S":
							self.isWaiting = True
							self.misses = 0
						self.lastState = state

			else:
				self.misses += 1
				if self.misses >= BlackjackSpeaker.missesRequired:
					self.lastState = state
					self.repeats = 0
					self.misses = 0

	def say(self, phrase):
		self.engine.say(phrase)
		if self.thread == None or not self.thread.isAlive():
			self.thread = Thread(target=self.engine.runAndWait)
			self.thread.start()
			return True
		return False

	def _verbosePhrase(self, state):
		phrase = ""
		#dealer = filter(lambda g:g.isDealer, state.groups)
		#player = filter(lambda g:not g.isDealer, state.groups)

		if len(state.dealers)!=1 or len(state.players)!=1:
			return phrase
		phrase += "I have "+str(state.players[0].score)
		phrase += " and dealer has "+str(state.dealers[0].score)+"; "
		return phrase

BlackjackSpeaker()