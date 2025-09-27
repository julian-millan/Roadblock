from abc import ABC, abstractmethod
import cv2
import os

class VLMClient(ABC):
    """ Base class for VLM clients.
    If implementing your own camera wrapper, you should inherit from this class and implement the following methods:
    - get_responses
    - get_next_response
    """

    def __init__(self):
        self.history = []

    @abstractmethod
    def get_responses(self, prompts, **kwargs) -> list:
        """ Returns a list of VLM responses for the given prompts. """
        pass

    @abstractmethod
    def get_next_response(self, next_prompt, **kwargs) -> str:
        """ Returns the next VLM response for the given prompt. """
        pass

    def set_output_root_path(self, path: str):
        """ Sets the root path where chat histories will be saved. """
        self.output_root_path = path
        os.makedirs(self.output_root_path, exist_ok=True)

    def save_chat(self, subdir_name: str, prompts, responses):
        """ Saves the chat history to a directory.
        Each prompt and response is saved in a separate text file, and images are saved as PNG files.
        Args:
            subdir_name (str): The directory where the chat history will be saved.
            prompts (list): A list of prompts, each prompt is a dictionary with keys "system", "user", and optionally "images".
            responses (list): A list of responses corresponding to the prompts.
        """
        os.makedirs(os.path.join(self.output_root_path, subdir_name), exist_ok=True)
        assert len(prompts) == len(responses)
        for i in range(len(prompts)):
            with open(os.path.join(self.output_root_path, subdir_name, f"prompt_{i+1:d}.txt"), "w") as file:
                if "system" in prompts[i] and len(prompts[i]["system"]) > 0:
                    file.write(prompts[i]["system"] + "\n" + prompts[i]["user"])
                else:
                    file.write(prompts[i]["user"])
            if not len(responses[i]) == 0:
                with open(os.path.join(self.output_root_path, subdir_name, f"response_{i+1:d}.txt"), "w") as file:
                    file.write(responses[i])
            if "images" in prompts[i]:
                for j in range(len(prompts[i]["images"])):
                    cv2.imwrite(os.path.join(self.output_root_path, subdir_name, f"image_{i+1:d}_{j+1:d}.png"), cv2.cvtColor(prompts[i]["images"][j], cv2.COLOR_RGB2BGR))

    def clear_history(self):
        """ Clears the internal chat history. """
        self.history = []