from abc import ABC, abstractmethod
from pydub import AudioSegment
import simpleaudio as sa

class TTSEngine(ABC):
    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        pass

    @abstractmethod
    def say(self, text: str):
        if self.is_speaking:
            self.stop()
        if self.stt_engine.running:
            self.stt_engine.stop_recognition()
            self.wait_for_speech_finish = True

    @abstractmethod
    def store(self, text: str, filename: str):
        pass

    @abstractmethod
    def stop(self):
        pass

    def get_duration(self, filename):
        audio = AudioSegment.from_wav(filename)
        return len(audio) / 1000 # original units in ms

    def play_audio(self, filename, blocking=False):
        if filename[-4:] != ".wav":
            raise RuntimeError(f"Expected file name to end in .wav, but got {filename}.")
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        if blocking:
            play_obj.wait_done()