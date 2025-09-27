import copy
import cv2
import numpy as np
import re
from typing import Optional

from prompts.drawing_utils import draw_axes, blur_threshold, white_tint, overlay_waypoints
from vision.utils import rotate
from trajectories import Trajectory

def get_position_prompt_single_segment(position_segment: list[np.ndarray], pause_segment: list[Optional[float]], waypoint_num=1):
    """ Generates the position description part of the prompt for a single image. """
    
    position_prompt = ""
    skip_point = False
    for i in range(len(position_segment)):
        if i != 0:
            if pause_segment[i] is not None and pause_segment[i] > 0:
                position_prompt += f"At waypoint {waypoint_num}, the robot stopped for {pause_segment[i]:.1f} seconds. "
                skip_point = True
            else:
                motion_direction = (position_segment[i] - position_segment[i-1]) * 100
                position_prompt += f"From waypoint {waypoint_num} to waypoint {waypoint_num+1}, the general motion of the robot is "
                position_prompt += f"({motion_direction[0]:.1f}, {motion_direction[1]:.1f}, {motion_direction[2]:.1f}) centimeters. "
                waypoint_num += 1
        if skip_point:
            skip_point = False
        else:
            gripper_pos_cart = position_segment[i] * 100
            position_prompt += f"At waypoint {waypoint_num}, the position of the gripper center is at ({gripper_pos_cart[0]:.1f}, {gripper_pos_cart[1]:.1f}, {gripper_pos_cart[2]:.1f}). "
    position_prompt += "\n"
    return position_prompt, waypoint_num

def get_velocity_prompt_single_segment(vel_segment: list[float], pause_segment: list[Optional[float]], waypoint_num=1):
    """ Generates the position description part of the prompt for a single image. """
    
    velocity_prompt = ""
    for i in range(1,len(vel_segment)):
        if pause_segment[i] is not None and pause_segment[i] > 0:
            velocity_prompt += f"At waypoint {waypoint_num}, the robot stopped for {pause_segment[i]:.1f} seconds. "
        else:
            avg_vel = vel_segment[i] * 100
            velocity_prompt += f"From waypoint {waypoint_num} to waypoint {waypoint_num + 1}, the average velocity of the robot is {avg_vel:.1f} centimeters per second. "
            waypoint_num += 1
    velocity_prompt += "\n"
    return velocity_prompt, waypoint_num

def get_force_prompt_single_segment(force_segment: list[list[float]], waypoint_num=1):
    """ Generates the force description part of the prompt for a single image. """
    
    force_prompt = ""
    for i in range(1,len(force_segment)):
        force_traj = force_segment[i]
        if np.sum(np.abs(force_traj)) > 0.1:
            force_prompt += f"From waypoint {waypoint_num} to waypoint {waypoint_num+1}, the force magnitude of the robot end-effector is: ["
            for force in force_traj:
                force_prompt += f"{force:.2f}, "
            force_prompt = force_prompt[:-2]
            force_prompt += "]."
        waypoint_num += 1
    if force_prompt == "":
        force_prompt += "There is no external force planned during this section. "
    force_prompt += "\n"
    return force_prompt, waypoint_num

def get_segment_prompt(trajectory_list: list[Trajectory], landmarks: None | dict = None):
    """ Generates the kinematic description for each image, so that the output is also a list
        # of images = length of each input = length of output """
    output_list = []
    waypoint_num = 1
    for i in range(len(trajectory_list)):
        # Initialization and gripper
        output_prompt = f"# Kinematic description for image {i+1}\n"
        output_prompt += f"At the start of image {i+1}, the gripper is {'open' if trajectory_list[i][0].gripper > 0 else 'closed'}.\n"
        # Position
        next_pos_prompt, _ = get_position_prompt_single_segment([w.position for w in trajectory_list[i]], [w.pause for w in trajectory_list[i]], waypoint_num)
        output_prompt += "## Position descriptions\n"
        output_prompt += next_pos_prompt
        # Velocity
        next_vel_prompt, _ = get_velocity_prompt_single_segment([w.velocity for w in trajectory_list[i]], [w.pause for w in trajectory_list[i]], waypoint_num)
        output_prompt += "## Velocity descriptions\n"
        output_prompt += next_vel_prompt
        next_force_prompt, waypoint_num = get_force_prompt_single_segment([w.force for w in trajectory_list[i]], waypoint_num)
        output_prompt += "## Force descriptions\n"
        output_prompt += next_force_prompt
        if landmarks != None:
            output_prompt += get_landmark_prompt(landmarks, [w.position for w in trajectory_list[i]])
        output_list.append(output_prompt)
    return output_list

def get_landmark_prompt(detected_landmarks, positions):
    landmark_output = ["left wrist", "right wrist",
                       "left elbow", "right elbow",
                       "left shoulder", "right shoulder",
                       "mouth"]
    output = "## Human body landmark positions\n\
The following are the positions for each of the detected body landmarks in the image: "
    start_point = positions[0] * 100
    end_point = positions[-1] * 100
    start_to_landmark = np.zeros(len(landmark_output))
    end_to_landmark = np.zeros(len(landmark_output))
    for i, joint_name in enumerate(landmark_output):
        joint_pos = None
        if joint_name != "mouth":
            if detected_landmarks[joint_name] is not None:
                joint_pos = detected_landmarks[joint_name] * 100
        else:
            if detected_landmarks["mouth (left)"] is not None and detected_landmarks["mouth (right)"] is not None:
                mouth_left = detected_landmarks["mouth (left)"] * 100
                mouth_right = detected_landmarks["mouth (right)"] * 100
                joint_pos = (mouth_left + mouth_right) / 2
        if joint_pos is not None:
            output += joint_name + f": ({joint_pos[0]:.1f}, {joint_pos[1]:.1f}, {joint_pos[2]:.1f}) centimeters, "
            start_to_landmark[i] = np.linalg.norm(start_point - joint_pos)
            end_to_landmark[i] = np.linalg.norm(end_point - joint_pos)
        else:
            start_to_landmark[i] = np.inf
            end_to_landmark[i] = np.inf
    output = output[:-2]
    output += ".\n"

    start_ind = np.argmin(start_to_landmark)
    end_ind = np.argmin(end_to_landmark)
    output += f"The first waypoint in this segment is closest to the {landmark_output[start_ind]} at a distance of {start_to_landmark[start_ind]:.1f} centimeters. "
    output += f"The last waypoint in this segment is closest to the {landmark_output[end_ind]} at a distance of {end_to_landmark[end_ind]:.1f} centimeters. "
    output += "\n"

    return output

def get_reasoning_prompt(previous_responses: list[str]) -> dict:
    stage_1_response = previous_responses[0]
    per_segment_descriptions = []
    for i in range(1, len(previous_responses)):
        pattern = r'(?ims)^##\s*segment description\s*:?\s*\n(.*?)(?=^\s*#|\Z)'
        match = re.search(pattern, previous_responses[i])
        per_segment_descriptions.append(match.group(1).strip())
    reasoning_prompt = "You are a robot, and here is what you currently observe in the environment:\n"
    reasoning_prompt += stage_1_response
    reasoning_prompt += "\n\n"
    reasoning_prompt += "You also have a trajectory planned to interact with the environment. It is broken down into segments, and here are the descriptions for each segment:\n"
    for i in range(len(per_segment_descriptions)):
        reasoning_prompt += f"- Segment {i+1}: {per_segment_descriptions[i]}\n"
    reasoning_prompt += "\n"
    pattern = r'(?ims)^##\s*overall intention\s*:?\s*\n(.*?)(?=^\s*#|\Z)'
    match = re.search(pattern, previous_responses[-1])
    reasoning_prompt += f"Additionally, from the motion and the observation, the intention of the entire trajectory is tentatively deduced to be: \"{match.group(1).strip()}\"\n\n"
    pattern = r'(?ims)^##\s*user cooperation\s*:?\s*\n(.*)'
    match = re.search(pattern, previous_responses[-1])
    reasoning_prompt += f"Further, the user cooperation in each segment needed in order to achieve the overall intention is tentatively deduced to be: \"{match.group(1).strip()}\"\n\n"
    reasoning_prompt += "I now need you to do the following: first rephrase the overall intention and user cooperation so that they are very friendly and suitable in an assistive setting. Don't use emotionless words like \"task\". Also ensure that these make sense within the context of each segment's description, and resolve any inconsistencies. Take the most likely intention and don't give ambiguous answers.\n\
Next, I need you to output one concise sentence per segment, as a statement that speaks to the user before you execute the motion for that segment. \
Include the overall **intention** where appropriate, and mention any active human cooperation required. Pay attention to when the robot actually plans to engage in contact, and make sure your statements are consistent with this--when the description for a segment says no force, it directly means that there is no contact, either the contact has not started yet, or the contact is already finished. \
Note that with the exception of the first segment, the statement will be said *while* the robot is moving, so it'll be the best approach to include the high-level intention in the first statement and omit it in the remaining statements. \
In terms of the motion, each statement should only talk about the motion within that segment. \
In summary, each of your sentences should contain the following information (in no particular order):\n\
- where the robot is moving to within the segment (starting position can in general be omitted, so no need for phrases like \"from <some place>\")\n\
- *If the shape of the trajectory in the segment is more complicated than a straight line*, then mention it, otherwise just say where you'll move to without saying \"in a straight line\"\n\
- *If velocity is not near-constant*, then changes and trends in velocity, and where they happen. If the description says the velocity/speed is constant/steady within the segment, don't mention anything about the speed all together.\n\
- *If there is any force*, then changes and trends in force, and where they happen. If the description says no force/pressure, it simply means that no contact is planned for the robot to actively engage with the environment, and in this case don't mention anything about force/contact all together.\n\
- *If needed*, any behavior the human should do in this segment to achieve the overall intention. Be specific for this one--you need to say **exactly** what the human should do to cooperate. If necessary, modify the previously determined user cooperation.\n\
- If appropriate, the intention of the segment, but be simple and basic\n\
Try to weave these different information together using simple language, and use qualitative descriptions instead of numerical values. Be concise and direct. \
For example, simply stating \"getting faster\" is an implication that you are starting slow, so no need to elaborate by saying things like \"starting slow and then speeding up\". \
If the description says something like \"no force\" or \"no contact\", it simply means that the robot is not planning to touch anything, so if the previous segment contains force and the current segment does not, it simply means that the contact is finiehd. Do not get this confused. \
You can ignore or condense extra information from the provided descriptions if they don't fit in one of the categories listed above. Suppose you are speaking to a ten-year-old. Simplify as much as you can. \
Speak in a friendly and helpful manner. Format your output as \"Statement x: <your sentence>\"."
    system_prompt = "We define \"intention\" as the overall task the robot is trying to accomplish, and \"motion\" as how the robot moves, including position, velocity, and force; these are two different things and it is very important to stick to this definition."
    structured_prompt = {
        "system": system_prompt,
        "user": reasoning_prompt
    }
    return structured_prompt

def get_vlm_prompts(segmented_data: dict) -> list[dict]:
    trajectory = segmented_data["trajectory"]
    gripper_pos_pixel = segmented_data["gripper_pos_pixel"]
    pose_camera_gripper = segmented_data["pose_camera_gripper"]
    camera_intrinsics = segmented_data["camera_intrinsics"]
    camera_extrinsics = segmented_data["T_base_camera_color"][0]
    detected_landmarks = segmented_data["landmark_pos"]

    threshold = (1.0 + np.max([np.linalg.norm(pose[0]) for pose in pose_camera_gripper])) * 1000
    partial_blurred = blur_threshold(segmented_data["initial_rgb"], segmented_data["initial_depth"], threshold)
    blurred_axes = draw_axes(partial_blurred, camera_intrinsics=camera_intrinsics, camera_extrinsics=camera_extrinsics)

    whitened_base = white_tint(partial_blurred, white_proportion=0.6)
    whitened_axes = draw_axes(whitened_base, camera_intrinsics=camera_intrinsics, camera_extrinsics=camera_extrinsics)

    robot_prompt = \
        "# General descriptions:\n\
You are a robot, where you have planned a trajectory to interact with what you see around you. You have a gripper that may hold an object. "
    environment_prompt = \
        "You are given two base images of what you currently see, with the first one being a big-picture view, and the second one from a camera mounted at the wrist of the gripper, \
capturing the content held by the gripper. For privacy purposes, all humans being captured in the picture have their face blurred out. However, all major body joints have been detected and labeled, \
as well as notable facial landmarks (eyes, nose, and mouth). Note that a light blue-colored landmark means it's the person's left side, and a yellow-colored landmark means the person's right side. Remember this in your reasoning. \
Observe, and respond to the following questions:\n\
    What are currently in the environment?\n"
    overlay_prompt = \
        "# Understanding your planned trajectory\n\
Your planned trajectory is represented both visually through images and textually. Below are explanations for the meaning of these representations.\n\
## Planned trajectory represented in images:\n\
Here is sequence of images showing your planned trajectory as overlays on the base image. "
    textual_prompt = \
        "## Overview of textual description to the planned trajectory:\n\
For each image in the sequence, there is textual description of the robot end effector's position, velocity, force, and other relevant information, mostly regarding the waypoints shown in the images. The meaning of these descriptions are explained below: "
    position_intro = \
        "### Position descriptions:\n\
These describe the position of the robot end-effector at each waypoint, as well as the relative translation between adjacent waypoints. Units are in centimeters."
    velocity_intro = \
        "### Velocity descriptions:\n\
These describe the average velocity of the robot end-effector as it goes between adjacent waypoints. Units are in centimeters per second."
    force_intro = \
        "### Force descriptions:\n\
These describe the force profile of the robot end-effector as it goes between adjacent waypoints, if there are moments of nonzero forces between two waypoints. \
Each nonzero profile is represented as a list, showing the change in force from one waypoint to the next. The first entry in the list represents the force at the starting waypoint, \
and the last entry represents the force at the ending waypoint. The middle entries represent the force profile as the robot end-effector moves from the starting waypoint to the ending waypoint. \
These forces represent the amount of force that the robot's end-effector is applying to some external object, and ignore the effect of anything that the robot is holding. \
This means that the robot plans to be in contact with some external object if and only if there are forces present. Units are in Newtons."
    gripper_prompt = \
        "### Gripper status\n\
You will be told the gripper state (open or closed) at the start of each image. It maintains the same throughout the same image, and may only change in between images. "
    axes_prompt = \
        "### Coordinate convention\n\
The coordinate system in these descriptions are defined relative to the robot's base, and an illustration of this convention is shown in the top left corner of every image, \
where red represents the positive x direction, green represents the positive y direction, and blue represents the positive z direction. \
The positive z direction is also the 'upwards' direction in real world, so it is upwards relative to any reference object that typically lies in the x-y plane (e.g. tables, floors)."
    if trajectory[0][0].gripper <= 0:
        environment_prompt += "    What is the robot holding in between its grippers?\n\
    Use your most likely guess for what the thing is, and don't infer beyond what you see. Suppose the robot does not let go of what it's holding. Then what function could this thing serve, in the context?\n"
        gripper_prompt += \
    "Note that the gripper is closed at the start, and whatever the robot is holding will continue to be held, unless there is further communication saying that the gripper has been opened. \
Therefore, while the gripper is holding the object, reason about the function of the object and how it might interact with the environment. "
    skeletal_model_intro = \
        "## Human body landmark positions\n\
In addition to the robot planned trajectory, you will also be told the position of each of the visible body landmarks of any human being captured in the image. \
For example, if an arm is visible, then you will be told the positions of the wrist, elbow, and shoulder. These positions are defined in the same coordinate system as the trajectory. \
Additionally, for each image, you will be told which landmark is closest to the starting and ending waypoint of the trajectory respectively, along with the distance to the waypoint. "

    overlay_prompt += "You will be shown the sequence one at a time, each accompanied by a cropped version that better displays the trajectory. "
    overlay_prompt += "Your planned trajectory is shown as waypoints connected by lines. "
    overlay_prompt += "In each image, the starting position is shown as a blue square, and the ending position as a red square. The rest of the waypoints in the middle are shown as circles in different shades of green, darker with lower velocities, and brighter with higher velocities. "
    overlay_prompt += "As another visual cue beyond the color, the length of the line segment connecting two consecutive waypoints is directly proportional to the speed at which the segment is travelled at. "
    overlay_prompt += "Additionally, forces are also represented in the images: The line segments connecting the waypoints are shown in cyan if the segment doesn't have any force. If there is force, then the line segment is colored in a gradient that shows cyan when the force is small and magenta when the force is large. These small and large are relative to the segment itself, and not absolute. "
    overlay_prompt += "Each subsequent image in the sequence starts at the last waypoint of the previous image and continues the trajectory. "
    overlay_prompt += "The base image is whitened for you to more clearly observe the overlay trajectories. "

    by_image_prompt = get_segment_prompt(trajectory_list=trajectory, landmarks=detected_landmarks)

    overlay_image_list = overlay_waypoints(whitened_axes, gripper_pos_pixel, [[w.velocity for w in segment] for segment in trajectory], [[w.force for w in segment] for segment in trajectory], circle_size=6, line_width=4)
    overlay_image_cropped_list = overlay_waypoints(whitened_axes, gripper_pos_pixel, [[w.velocity for w in segment] for segment in trajectory], [[w.force for w in segment] for segment in trajectory], circle_size=6, crop=True)

    task_intro = \
        "# Task descriptions\n\
Your high-level goal is to summarize what the planned trajectory of the robot is trying to accomplish, and do so without numerical values, but with references to other components of the environment. You will answer a series of questions as you move towards building a complete statement.\n"
    task_prompt_1 = \
    "For each image (segment), you answer the following questions:\n\
1) Consider the position of the robot end-effector. Answer the following primarily from looking at the images, but refer to human body landmarks in the exact name (left shoulder, right elbow, etc.). When describing left and right body parts, always reference them from the person's perspective and not from the image. \
Light blue landmarks indicate the person's left side, and light yellow landmarks indicate the person's right side. Do not mix them up. This is regardless of image orientation. Use the trajectory in the image as your top reference, and use the cropped version for you to more clearly observe the trajectory itself. Do not make additional assumptions.\n\
    1a) Where is the blue square waypoint in the image? (this is where the motion starts) How close is the starting point in this segment to the nearest body waypoint, by looking at the textual information provided? If it is close, then the blue waypoint you see should be near that landmark in the image as well. Note that this is often the ending position of the last segment, unless this is the first segment, in which case it should be near the robot's gripper. Also note that this is different from the light blue body landmark. The landmarks are larger and have a paler color.\n\
    1b) Where is the red square waypoint in the image? (this is where the motion ends) How close is the ending point in this segment to the nearest body waypoint, by looking at the textual information provided? If it is close, then the red waypoint you see should be near that landmark in the image as well.\n\
    1c) What is the shape of the trajectory? Is it a straight line, or some other shape?\n\
    1d) Does the trajectory get close to some reference object in the environment, or pass through any human landmark? This can be either in the middle of the motion or at the end. If so, note the waypoint at which the robot gets closest to any landmarks/objects. Note that the trajectory may not be a straight line, so be sure to trace all the points from the start to the end, along the line segments.\n\
    1e) Is it moving towards something in the environment? If not, then is it moving in some nominal direction? You may use 'upward' and 'downward' to indicate +z and -z axes, as well as 'forward' and 'backward' to indicate the robot arm's reaching out and retracting back.\n\
2) Consider the velocity of the robot end-effector. Recall that a brighter-green waypoint indicates higher speed, and darker-green point means lower speed. Length of line segments connecting two waypoints is also directly proportional to the speed. \
Base your response primarily on the textual velocity descriptions, and only use visual when you want to locate an interesting waypoint. \
Note that for this part, we define a \"notable difference\" or a \"change\" as anything greater than a two times difference. So if the speed decreases to more than half, it's called a slowdown, and if the speed increases to more than twice before, it's called a speedup.\n\
    2a) What is the speed of the robot near the start of this segment?\n\
    2b) What is the speed of the robot near the end of this segment?\n\
    2c) Is the starting speed notably different from the ending speed of the last segment?\n\
    2d) Is the ending speed in this segment notably different from the starting speed in this segment?\n\
    2e) If there is a notable change in velocity in this segment, either within the segment itself or as it transitions into this segment from the last, answer additionally the following:\n\
        2e.i) At which waypoint does this change happen?\n\
        2e.ii) Where is the robot end-effector position located within the environment, at this waypoint? Refer back to question 1 and the images to answer this.\n\
        2e.iii) After this change, where is the robot moving towards?\n\
3) Consider the force of the robot end effector. Recall that the line segments between two waypoints are colored according to the external forces as the end effector travels through that segment. In a gradient from blue to red, blue means lower force and red means higher force.\n\
    3a) Are there forces involved? If not, then you don't have to answer the rest of the questions in question group 3.\n\
    3b) If so, in what region is this force being applied on? Refer back to question 1 on the positions, as well as the images to answer this.\n\
    3c) Is that force changing throughout the segment? If so, how is it changing over the region that the force is applied on?"
    held_object_force_prompt = "\
    3d) The robot gripper is holding something in this segment; it is likely that the forces are being applied through this object being held, so what function could the held object serve, in relation to other components in the environment? What is robot planning to interact with?"
    task_prompt_2 = "\
4) Consider all of position, velocity, and force in this segment.\n\
    4a) What is the intention of the motion in this segment? What did you recognize the object between the grippers to be? Infer about its functionality given the position, velocity, and force in this segment. Again, keep in mind that the robot plans to actively make contact if and only if there is nonzero force in this segment.\n\
## Overall description for each segment:\n\
After you answer these questions, associate the different components of this segment of motion---position, velocity, force---in the context of the environment. Generate a description of the robot's motion throughout this segment. \
When describing the motion direction, only use nominal directions when there isn't a clear object in the environment which the robot is moving to, or when the motion towards that object isn't a straight path. \
Otherwise, always describe the motion by referring to objects in the environment. If the motion is towards a body landmark, always use the exact name of the landmark, and don't refer to general body locations. The point is to be clear, specific, but also concise. \
Recall all the velocity and force changes that you noted. Where do these happen? Is it at a certain point or gradually? Include where the changes happen if there is a clear position. Keep your answer to this section under the heading \"## Segment description\". Do not use this heading anywhere else in responding to previous numbered questions."
    task_prompt_end = "## Overall intention for entire trajectory:\n\
What is the overall intention of the robot, now that you have seen all the planned trajectories? This should be closely related to what you identified that the robot is holding between the grippers. Reason about its functionality within the trajectory. Focus on the intention and not the motion. Keep your answer to this question under the heading \"## Overall intention\".\n\
Additionally, what does the user need to do in each segment that will allow this overall intention to be achieved? It is fine if there is no cooperation needed during some/all segments. Keep your answer to this question under the heading \"## User cooperation\".\n"

    full_prompt = []
    full_prompt.append({"system": robot_prompt,
                        "user": environment_prompt,
                        "images": [cv2.flip(rotate(blurred_axes, segmented_data["head_rotation"]),1), rotate(segmented_data["wrist_rgb"], segmented_data["wrist_rotation"])]})
    for i in range(len(by_image_prompt)):
        if trajectory[i][0].gripper <= 0:
            user_prompts = [by_image_prompt[i], task_intro, task_prompt_1, held_object_force_prompt, task_prompt_2]
        else:
            user_prompts = [by_image_prompt[i], task_intro, task_prompt_1, task_prompt_2]
        if i == len(by_image_prompt)-1:
            user_prompts.append(task_prompt_end)
        full_prompt.append({"system": "\n".join([overlay_prompt, textual_prompt, position_intro, velocity_intro, force_intro, axes_prompt, skeletal_model_intro, gripper_prompt]),
                            "user": "\n".join(user_prompts),
                            "images": [cv2.flip(rotate(overlay_image_list[i], segmented_data["head_rotation"]),1), cv2.flip(rotate(overlay_image_cropped_list[i], segmented_data["head_rotation"]),1)]})
    return full_prompt