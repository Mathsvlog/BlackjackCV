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
		self.engine.setProperty("rate", 120)
		voices = self.engine.getProperty("voices")
		self.engine.setProperty("voice", voices[-1].id)

		self.lastState = None# current belief state of the speaker
		self.repeats = 0# if hits repeatsRequired, will speak move
		self.misses = 0# if hits missresRequired, will set lastState to current state
		self.thread = None# the separate thread handling speaking
		self.isWaiting = False

	def analyzeState(self, state):
		print self.isWaiting
		if self.isWaiting:
			pass
			
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
					self.isWaiting = player.move == "S"
					if not self.say(phrase):
						self.repeats -= 1
		else:
			self.misses += 1
			if self.misses > BlackjackSpeaker.missesRequired:
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
		dealer = filter(lambda g:g.isDealer, state.groups)
		player = filter(lambda g:not g.isDealer, state.groups)
		if len(dealer)!=1 or len(player)!=1:
			return phrase
		phrase += "I have "+str(player[0].score)
		phrase += " and dealer has "+str(dealer[0].score)+"; "
		return phrase

BlackjackSpeaker()