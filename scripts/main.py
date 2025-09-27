import argparse
import cv2
import datetime
import numpy as np
import os
from pathlib import Path
import re
import time

from speech.baseline_communication import get_baseline_communication
from save_load import save_data, load_system
from vision.utils import rotate
from motion_planners import BathingPlanner, FeedingPlanner, ShavingPlanner
from vision.pose_detector import PoseDetector
from prompts.prompt_generator import get_vlm_prompts, get_reasoning_prompt
from trajectories.utils import segment_trajectory

LEFT_ARM = True # If False, use right arm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bathing", action="store_true")
    parser.add_argument("--feeding", action="store_true")
    parser.add_argument("--shaving", action="store_true")
    parser.add_argument("--trial", default=0, type=int)
    parser.add_argument("--no_comm", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("-p", default=None, type=str)
    args = parser.parse_args()

    repo_dir = Path(__file__).parent.parent

    if args.bathing:
        config_path = os.path.join(repo_dir, "configs", "bathing_stretch.json")
    elif args.feeding:
        config_path = os.path.join(repo_dir, "configs", "feeding_xarm.json")
        # config_path = os.path.join(repo_dir, "configs", "feeding_stretch.json")
    elif args.shaving:
        config_path = os.path.join(repo_dir, "configs", "shaving_stretch.json")
    
    robot, head_camera, wrist_camera, tts_engine, vlm_client = load_system(config_path)
    detector = PoseDetector()

    # Fetches and displays camera stream until enter is pressed
    cv2.namedWindow("Head camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wrist camera", cv2.WINDOW_NORMAL)
    while True:
        head_rgb, head_depth = head_camera.get_new_frames()
        wrist_rgb = wrist_camera.get_new_frames()
        cv2.imshow("Head camera", cv2.cvtColor(rotate(head_rgb, head_camera.rotation), cv2.COLOR_RGB2BGR))
        cv2.imshow("Wrist camera", cv2.cvtColor(rotate(wrist_rgb, wrist_camera.rotation), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == 13:
            break
    cv2.destroyAllWindows()

    start_time = time.perf_counter()

    # Detect landmarks, face blurring, and annotate landmarks
    detector.detect_landmarks(head_rgb, rotation=head_camera.rotation)
    blurred_image = detector.face_blurring(head_rgb) # original orientation
    annotated_image = detector.draw_landmarks_on_image(blurred_image) # original orientation
    depth_image = head_camera.depth_map_from_image(head_depth)
    combined_images = np.concatenate((cv2.cvtColor(rotate(annotated_image, head_camera.rotation), cv2.COLOR_RGB2BGR),
                                      cv2.cvtColor(rotate(depth_image, head_camera.rotation), cv2.COLOR_RGB2BGR)), axis=1)
    os.makedirs(os.path.join(repo_dir, "temp"), exist_ok=True)
    cv2.imwrite(os.path.join(repo_dir, "temp", "head_combined.png"), combined_images)
    cv2.imwrite(os.path.join(repo_dir, "temp", "wrist.png"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
    
    gripper_value = robot.get_gripper_value()
    camera_transform = robot.get_camera_transform()
    camera_intrinsics = head_camera.get_color_intrinsics()

    output_path = os.path.join(repo_dir, "output")
    if args.p is not None:
        output_path = os.path.join(output_path, "p_" + args.p)
    print(f"Initialization time: {time.perf_counter() - start_time:.3f}s.")

    # plan for corresponding task and trial
    start_time = time.perf_counter()
    if args.bathing:
        output_path = os.path.join(output_path, f'bathing_{args.trial}')
        planner = BathingPlanner()
        trajectory = planner.plan_trajectory(robot=robot, camera=head_camera, detector=detector, side="left" if LEFT_ARM else "right", trajectory_num=args.trial)
    elif args.feeding:
        output_path = os.path.join(output_path, f'feeding_{args.trial}')
        planner = FeedingPlanner()
        trajectory = planner.plan_trajectory(robot=robot, camera=head_camera, detector=detector, trajectory_num=args.trial)
    elif args.shaving:
        output_path = os.path.join(output_path, f'shaving_{args.trial}')
        planner = ShavingPlanner()
        trajectory = planner.plan_trajectory(robot=robot, camera=head_camera, detector=detector, side="left" if LEFT_ARM else "right", trajectory_num=args.trial)
    print(f"High-level planner time: {time.perf_counter() - start_time:.3f}s.")
    
    # densify trajectory through robot low-level controller
    start_time = time.perf_counter()
    dense_trajectory = robot.plan_low_level_trajectory(trajectory, camera=head_camera)
    print(f"Low-level planner time: {time.perf_counter() - start_time:.3f}s.")

    # query landmark positions
    keypoint_pos = detector.query_all_landmarks_3d(camera=head_camera, transform=camera_transform)

    if args.baseline:
        output_path += "_baseline"
    elif args.no_comm:
        output_path += "_no_comm"

    datetime_obj = datetime.datetime.now()
    output_path += f"_{datetime_obj.month}.{datetime_obj.day}_{datetime_obj.hour}:{datetime_obj.minute}"
    vlm_client.set_output_root_path(output_path)
    
    data = save_data(directory=output_path, filename="data.pkl", trajectory=dense_trajectory, landmark_pos=keypoint_pos, camera_transform=camera_transform, camera_intrinsics=camera_intrinsics,
                     raw_rgb=head_rgb, annotated_rgb=annotated_image, depth=head_depth, rgb_wrist=wrist_rgb, head_rotation=head_camera.rotation, wrist_rotation=wrist_camera.rotation)

    segmented_data = segment_trajectory(data)
    print(len(segmented_data["trajectory"]), type(segmented_data["trajectory"][0]))
    segment_timestamps = [segmented_data["trajectory"][i][-1].timestamp for i in range(len(segmented_data["trajectory"])-1)]
    vlm_prompts = get_vlm_prompts(segmented_data)

    if args.no_comm:
        time.sleep(20.0)
        vlm_client.save_chat("prompt_only", vlm_prompts, [""] * len(vlm_prompts))
    elif args.baseline:
        time.sleep(30.0)
        statements = get_baseline_communication(segmented_data)
        vlm_client.save_chat("prompt_only", vlm_prompts, [""] * len(vlm_prompts))
    else:
        responses = vlm_client.get_responses(vlm_prompts)
        reasoning_prompt = get_reasoning_prompt(responses)
        reasoning_response = vlm_client.get_next_response(reasoning_prompt, model="o3-mini", temperature=None, reasoning_effort="high", custom_history=[])

        vlm_prompts.append(reasoning_prompt)
        responses.append(reasoning_response)
        vlm_client.save_chat("full_chat", vlm_prompts, responses)
        
        print(f"VLM time: {time.perf_counter() - start_time:.3f}s.")
        start_time = time.perf_counter()

        response = reasoning_response.replace("*", "") # get rid of any bold font placed by VLM
        pattern = r"Statement\s*\d+:\s*([^\n]+)"
        statements = re.findall(pattern, response)            

    if not args.no_comm:
        for i in range(len(statements)):
            statement = statements[i]
            tts_engine.store(statement, os.path.join(output_path, f"statement_{i+1:d}.wav"))
        durations = [tts_engine.get_duration(os.path.join(output_path, f"statement_{i+1:d}.wav")) for i in range(len(statements))]

        print(f"TTS time: {time.perf_counter() - start_time:.3f}s.")
    
        statement_timestamps = np.array(segment_timestamps) - np.array(durations[1:])/2
        tts_engine.play_audio(os.path.join(output_path, f"statement_1.wav"), blocking=True)
        start_time = time.perf_counter()
        curr_statement_idx = 2
    
    try:
        robot.execute_trajectory(dense_trajectory, camera=head_camera)
        print(f"Execution time: {time.perf_counter() - start_time:.3f}s.")
        start_time = time.perf_counter()
        while robot.is_trajectory_active:
            if not args.no_comm and curr_statement_idx < len(statement_timestamps) + 2 and time.perf_counter() + 0.01 - start_time > statement_timestamps[curr_statement_idx-2]:
                tts_engine.play_audio(os.path.join(output_path, f"statement_{curr_statement_idx:d}.wav"))
                curr_statement_idx += 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        robot.stop()
        
    head_camera.close()
    wrist_camera.close()