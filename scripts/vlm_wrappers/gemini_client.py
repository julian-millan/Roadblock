import base64
import cv2
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.types import Content

from . import VLMClient

class GeminiClient(VLMClient):
    """ A wrapper for interacting with Google Gemini's API.
    This client allows sending prompts to Gemini models and receiving responses.
    It supports image inputs and maintains an internal chat history that is automatically updated with each interaction.
    """
    def __init__(self, api_key):
        """ Initializes the Gemini client with the provided API key. """
        self.client = genai.Client(api_key=api_key)
        self.history = []

    def encode_image_from_np(self, image_np):
        """ Encodes a numpy array image to a base64 string. 
        Args:
            image_np (numpy.ndarray): The image in numpy array format, expected to be in RGB format.
        Returns:
            The base64 encoded string of the image.
        """
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    def get_next_response(self, next_prompt, model="gemini-2.5-flash-preview-05-20", temperature=1, custom_history: Optional[list]=None) -> str:
        """ Returns the next VLM/LLM response for the given prompt.
        Args:
            next_prompt (dict): A dictionary containing the prompt with keys "system", "user", and optionally "images".
            model (str): The model to use for the response. Default is "gemini-2.5-flash-preview-05-20". Also supports "gemini-2.0-flash".
            temperature (float): The temperature for the response generation. Default is 1.
            custom_history (Optional[list]): If provided, this list will be used as the chat history instead of the internal history.
        Returns:
            The response from the VLM/LLM.
        """
        new_message = Content(role="user", parts=[{"text": next_prompt["user"]}])
        if custom_history is not None:
            custom_history.append(new_message)
            contents = custom_history
        else:
            self.history.append(new_message)
            contents = self.history
        config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        if len(next_prompt["system"]) > 0:
            config.system_instruction = next_prompt["system"]
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        if custom_history is not None:
            custom_history.append(Content(role="model", parts=[{"text": response.text}]))
        else:
            self.history.append(Content(role="model", parts=[{"text": response.text}]))
        print(response.text)
        return response.text

    def get_responses(self, prompts) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.get_next_response(prompt)
            responses.append(response)
        return responses
    
    def clear_history(self):
        """ Clears the internal chat history. """
        self.history = []
