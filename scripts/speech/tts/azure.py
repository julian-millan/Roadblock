import azure.cognitiveservices.speech as speechsdk
import time

from . import TTSEngine

class AzureTTSEngine(TTSEngine):
    def __init__(self, key, region):
        self.speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        self.speech_config.speech_synthesis_voice_name = "en-US-NancyMultilingualNeural"
        self._is_speaking = False
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def say(self, text: str):
        super().say(text)
        self.engine = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        self.engine.synthesis_completed.connect(self._speak_finished)
        self.engine.speak_text_async(text)
        self._is_speaking = True

    def _speak_finished(self, event):
        self._is_speaking = False
        if self.wait_for_speech_finish:
            self.wait_for_speech_finish = False
            self.stt_engine.start_recognition()

    def store(self, text: str, filename: str):
        if filename[-4:] != ".wav":
            raise RuntimeError(f"Expected file name to end in .wav, but got {filename}.")
        audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
        self.engine = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)
        self.engine.speak_text_async(text).get()
    
    def stop(self):
        self.engine.stop_speaking_async()
