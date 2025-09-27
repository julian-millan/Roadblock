import base64
import cv2
from typing import List, Optional
from openai import OpenAI

from . import VLMClient

class ChatGPTClient(VLMClient):
    """ A wrapper for interacting with OpenAI's API.
    This client allows sending prompts to OpenAI's models and receiving responses.
    It supports image inputs and maintains an internal chat history that is automatically updated with each interaction.
    """
    def __init__(self, api_key):
        """ Initializes the ChatGPT client with the provided API key. """
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def encode_image_from_np(self, image_np):
        """ Encodes a numpy array image to a base64 string. 
        Args:
            image_np (numpy.ndarray): The image in numpy array format, expected to be in RGB format.
        Returns:
            The base64 encoded string of the image.
        """
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    def get_next_response(self, next_prompt, model="gpt-4o", temperature=0.4, reasoning_effort="minimal", verbosity="low", custom_history: Optional[list]=None) -> str:
        """ Returns the next VLM/LLM response for the given prompt.
        Args:
            next_prompt (dict): A dictionary containing the prompt with keys "system", "user", and optionally "images".
            model (str): The model to use for the response. Can be any valid OpenAI model string in the family of "gpt-4", "gpt-5", "o3", or "o4".
            temperature (float): The temperature for the response generation. Default is 0.4. Ignored for o-series models.
            reasoning_effort (str): The reasoning effort for the response generation. Default is "minimal". Only used for reasoning models. Note that for o-series models, the default value is not supported.
            verbosity (str): The verbosity level for the response generation. Default is "low". Only used if model is "gpt-5".
            custom_history (Optional[list]): If provided, the internal history will be replaced with this list.
        Returns:
            The response from the VLM/LLM.
        """
        new_messages = []
        if len(next_prompt["system"]) > 0:
            new_messages.append({
                "role": "developer",
                "content": next_prompt["system"]
            })
        user_content = [{"type": "text", "text": next_prompt["user"]}]
        if "gpt" in model:
            if "images" in next_prompt and len(next_prompt["images"]) > 0:
                for image in next_prompt["images"]:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_from_np(image)}"}})
        new_messages.append({
            "role": "user",
            "content": user_content
        })
        if custom_history is not None:
            self.history = custom_history
        self.history.extend(new_messages)
        messages = self.history
        if "gpt-5" in model:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )
        elif "gpt-4" in model:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        elif "o3" in model or "o4" in model:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        self.history.append({"role": "assistant", "content": completion.choices[0].message.content})
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def get_responses(self, prompts) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.get_next_response(prompt)
            responses.append(response)
        return responses