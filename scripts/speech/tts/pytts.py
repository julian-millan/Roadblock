import pyttsx3

from . import TTSEngine

class PyTTSEngine(TTSEngine):
    def __init__(self, rate=120):
        self.rate = rate
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('voice', 24)
        self.engine.connect('finished-utterance', self._speak_finished)
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def say(self, text: str):
        super().say(text)
        self.engine.say(text)
        self.engine.startLoop(False)
        self._is_speaking = True

    def _speak_finished(self, completed):
        self._is_speaking = False
        if self.wait_for_speech_finish:
            self.wait_for_speech_finish = False
            self.stt_engine.start_recognition()

    def store(self, text: str, filename: str):
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()

    def stop(self):
        self.engine.stop()