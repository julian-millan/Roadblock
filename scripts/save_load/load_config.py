import importlib
import json
import numpy as np

from vision.cameras import BaseCamera
from robots import BaseRobot
from speech.tts import TTSEngine
from vlm_wrappers import VLMClient

def load_system(filename: str) -> tuple[BaseRobot, BaseCamera, BaseCamera, TTSEngine, VLMClient]:
    with open(filename, "r") as f:
        data = json.load(f)

    if data["robot"]["type"] == "XArmRobot":
        if "tcp_transform" in data["robot"]:
            tcp_transform = data["robot"]["tcp_transform"]
        else:
            tcp_transform = np.eye(4)
        from robots import XArmRobot
        robot = XArmRobot(data["robot"]["IP"], data["transform"], tcp_transform)
    elif data["robot"]["type"] == "StretchRobot":
        from robots import StretchRobot
        if "tool" in data["robot"]:
            robot = StretchRobot(tool=data["robot"]["tool"])
        else:
            robot = StretchRobot()
        robot.robot.end_of_arm.get_joint("wrist_yaw").motor.set_current_limit(60)

    if "rotation" in data["head_camera"]:
        head_rotation = resolve_dotted_path(data["head_camera"]["rotation"])
    else:
        head_rotation = None
    
    if "rotation" in data["wrist_camera"]:
        wrist_rotation = resolve_dotted_path(data["wrist_camera"]["rotation"])
    else:
        wrist_rotation = None

    if data["head_camera"]["type"] == "RealSenseCamera":
        options = dict()
        if "exposure" in data["head_camera"]:
            options["exposure"] = data["head_camera"]["exposure"]
        if "gain" in data["head_camera"]:
            options["gain"] = data["head_camera"]["gain"]
        from vision.cameras import RealSenseCamera
        head_camera = RealSenseCamera(data["head_camera"]["serial_number"], data["head_camera"]["model"], head_rotation, options)
    elif data["head_camera"]["type"] == "ZedMiniCamera":
        from vision.cameras import ZedMiniCamera
        head_camera = ZedMiniCamera(head_rotation)

    if data["wrist_camera"]["type"] == "RealSenseCamera":
        options = dict()
        if "exposure" in data["wrist_camera"]:
            options["exposure"] = data["wrist_camera"]["exposure"]
        if "gain" in data["wrist_camera"]:
            options["gain"] = data["wrist_camera"]["gain"]
        from vision.cameras import RealSenseCamera
        wrist_camera = RealSenseCamera(data["wrist_camera"]["serial_number"], data["wrist_camera"]["model"], wrist_rotation, options)
    elif data["wrist_camera"]["type"] == "ZedMiniCamera":
        from vision.cameras import ZedMiniCamera
        wrist_camera = ZedMiniCamera(wrist_rotation)

    if data["text_to_speech"]["type"] == "AzureTTSEngine":
        from speech.tts import AzureTTSEngine
        tts_engine = AzureTTSEngine(data["text_to_speech"]["key"], data["text_to_speech"]["region"])
    elif data["text_to_speech"]["type"] == "PyTTSEngine":
        from speech.tts import PyTTSEngine
        if "rate" in data["text_to_speech"]:
            tts_engine = PyTTSEngine(data["text_to_speech"]["rate"])
        else:
            tts_engine = PyTTSEngine()

    if data["VLM"]["type"] == "ChatGPT":
        from vlm_wrappers import ChatGPTClient
        vlm_client = ChatGPTClient(data["VLM"]["key"])

    return robot, head_camera, wrist_camera, tts_engine, vlm_client

def resolve_dotted_path(dotted_path: str):
    """
    Resolves a dotted string like 'cv2.ROTATE_180' to the actual Python object.
    """
    parts = dotted_path.split('.')
    module_path = parts[0]
    try:
        # Load the base module first
        obj = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module {module_path}") from e

    # Traverse the rest of the path
    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise AttributeError(f"Module '{module_path}' has no attribute '{part}'") from e
    return obj
