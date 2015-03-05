import pyttsx
from threading import Thread


class BlackjackSpeaker:

	repeatsRequired = 5
	phrases = {"H": "hit me", "S":"stand", "P":"split the cards", "D":"double down", "B":"i bust"}

	def __init__(self):
		self.engine = pyttsx.init()
		self.engine.setProperty("rate", 150)
		voices = self.engine.getProperty("voices")
		self.engine.setProperty("voice", voices[0].id)
		self.lastState = None
		self.repeats = 0
		self.thread = None

	def analyzeState(self, state):
		if state == self.lastState:
			self.repeats += 1
			if self.repeats == BlackjackSpeaker.repeatsRequired and state.groups!=None:
				# figure out move
				player = filter(lambda group:not group.isDealer, state.groups)[0]
				if player.move in BlackjackSpeaker.phrases:
					if not self.say(BlackjackSpeaker.phrases[player.move]):
						self.repeats -= 1
		else:
			self.lastState = state
			self.repeats = 0

	def say(self, phrase):
		self.engine.say(phrase)
		if self.thread == None or not self.thread.isAlive():
			self.thread = Thread(target=self.engine.runAndWait)
			self.thread.start()
			return True
		return False


BlackjackSpeaker()