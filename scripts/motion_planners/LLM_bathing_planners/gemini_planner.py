def plan_trajectory(self, robot: BaseRobot, camera: BaseCamera, detector: PoseDetector, side="left", trajectory_num=1) -> list[Waypoint]:
    fk = robot.forward_kinematics()
    camera_transform = robot.get_camera_transform()
    gripper_value = robot.get_gripper_value()
    start_point = fk[0:3,3]
    start_orientation = fk[0:3,0:3]
    start_up = start_point + np.array([0,0,0.1])
    wrist_pixel = detector.query_landmark(side + " wrist")
    # elbow_pixel = detector.query_landmark(side + " elbow")
    # Gemini, the above code should remain the same. Add the roadblock pixels explicitly below.
    # Assuming 'side' is 'right' to match the arm with roadblocks in the image.
    # Pixel coordinates are estimated from the image (approx. 275w x 475h).
    elbow_pixel = np.array([160, 360]) # Estimated elbow pixel
    # watch_pixel = np.array([120, 410]) # Watch is at the wrist, will assume wrist_pixel is start
    bandage_lower_pixel = np.array([140, 320]) # Estimated bandage pixel (edge near wrist)
    bandage_upper_pixel = np.array([150, 300]) # Estimated bandage pixel (edge near elbow)
    # Gemini, you need to get the elbow pixel from the image (dont query landmarks), then transform the roadblock pixels to points below
    wrist_point = camera.get_3d_from_pixel(pixel=wrist_pixel, transform=camera_transform)
    elbow_point = camera.get_3d_from_pixel(pixel=elbow_pixel, transform=camera_transform)
    # Transform roadblock pixels to 3D points
    bandage_lower_point = camera.get_3d_from_pixel(pixel=bandage_lower_pixel, transform=camera_transform)
    bandage_upper_point = camera.get_3d_from_pixel(pixel=bandage_upper_pixel, transform=camera_transform)
    

    # This next code block should remain unchanged
    forearm_dir = (elbow_point - wrist_point) / np.linalg.norm(elbow_point - wrist_point)
    path_pixels = np.linspace(wrist_pixel, elbow_pixel, 10, dtype=np.uint16, endpoint=True)
    path_normals = [camera.get_normal_from_pixel(pixel=pixel, transform=camera_transform) for pixel in path_pixels]
    bathe_start_orientation = self.bathing_orientation_fn(forearm_dir)(np.mean(path_normals, axis=0))
    

    if trajectory_num == 1:
        elbow_up = elbow_point + np.array([0,0,0.1])
        wrist_up = wrist_point + np.array([0,0,0.04])
        # Reason about potential roadblock_up points here, if needed
        # Define "up" points to navigate over the bandage, using a 5cm offset
        bandage_lower_up = bandage_lower_point + np.array([0,0,0.05])
        bandage_upper_up = bandage_upper_point + np.array([0,0,0.05])

        trajectory = Trajectory()
        # Add trajectory waypoints to navigate any potential roadblocks, be sure to integrate it smoothly into the trajectory
        trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
        trajectory.add_waypoint(Waypoint(wrist_up, rotation=bathe_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
        # Assuming wrist_point is a safe start, just below the watch
        trajectory.add_waypoint(Waypoint(wrist_point, rotation=bathe_start_orientation, velocity=0.02, force=[], gripper=gripper_value))
        # 1. Wash from wrist to just before the bandage
        trajectory.add_waypoint(Waypoint(bandage_lower_point, rotation=None, velocity=0.01, force=[1.0, 3.0], gripper=gripper_value, surface_align_fn=self.bathing_orientation_fn(forearm_dir), start_pixel=wrist_pixel, end_pixel=bandage_lower_pixel))
        # 2. Lift up to avoid bandage
        trajectory.add_waypoint(Waypoint(bandage_lower_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
        # 3. Move over the bandage
        trajectory.add_waypoint(Waypoint(bandage_upper_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
        # 4. Move down after the bandage
        trajectory.add_waypoint(Waypoint(bandage_upper_point, rotation=bathe_start_orientation, velocity=0.02, force=[], gripper=gripper_value))
        # 5. Wash from after the bandage to the elbow
        trajectory.add_waypoint(Waypoint(elbow_point, rotation=None, velocity=0.01, force=[1.0, 3.0], gripper=gripper_value, surface_align_fn=self.bathing_orientation_fn(forearm_dir), start_pixel=bandage_upper_pixel, end_pixel=elbow_pixel))
        # 6. Lift off from elbow and return
        trajectory.add_waypoint(Waypoint(elbow_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
        trajectory.add_waypoint(Waypoint(start_up, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
        trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.04, force=[], gripper=gripper_value))

    elif trajectory_num == 2:
        # Reason about potential roadblocks around the shoulder add repeat your pixel and point inclusions while including these new roadblocks
        # Re-define roadblock pixels and points as they are local to the 'if' block
        elbow_pixel = np.array([160, 360])
        bandage_lower_pixel = np.array([140, 320])
        bandage_upper_pixel = np.array([150, 300])

        # Transform roadblock pixels to 3D points
        bandage_lower_point = camera.get_3d_from_pixel(pixel=bandage_lower_pixel, transform=camera_transform)
        bandage_upper_point = camera.get_3d_from_pixel(pixel=bandage_upper_pixel, transform=camera_transform)
        
        # Define "up" points for roadblocks
        bandage_lower_up = bandage_lower_point + np.array([0,0,0.05])
        bandage_upper_up = bandage_upper_point + np.array([0,0,0.05])

        shoulder_pixel = detector.query_landmark(side + " shoulder")
        shoulder_point = camera.get_3d_from_pixel(pixel=shoulder_pixel, transform=camera_transform)
        shoulder_normal = camera.get_normal_from_pixel(pixel=shoulder_pixel, transform=camera_transform)
        shoulder_up = shoulder_point + shoulder_normal * 0.1
        wrist_up = wrist_point + np.array([0,0,0.04])

        trajectory = Trajectory()
        # Add trajectory waypoints to navigate any potential roadblocks, be sure to integrate it smoothly into the trajectory
        trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
        trajectory.add_waypoint(Waypoint(wrist_up, rotation=bathe_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
        # 1. Start at wrist (assuming safe)
        trajectory.add_waypoint(Waypoint(wrist_point, rotation=bathe_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
        # 2. Wash from wrist to lower bandage
        trajectory.add_waypoint(Waypoint(bandage_lower_point, rotation=None, velocity=0.01, force=[1.0, 1.0], gripper=gripper_value, surface_align_fn=self.bathing_orientation_fn(forearm_dir), start_pixel=wrist_pixel, end_pixel=bandage_lower_pixel))
        # 3. Lift up
        trajectory.add_waypoint(Waypoint(bandage_lower_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
        # 4. Move over
        trajectory.add_waypoint(Waypoint(bandage_upper_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
        # 5. Move down
        trajectory.add_waypoint(Waypoint(bandage_upper_point, rotation=bathe_start_orientation, velocity=0.02, force=[], gripper=gripper_value))
        # 6. Wash from upper bandage to elbow
        trajectory.add_waypoint(Waypoint(elbow_point, rotation=None, velocity=0.01, force=[1.0, 1.0], gripper=gripper_value, surface_align_fn=self.bathing_orientation_fn(forearm_dir), start_pixel=bandage_upper_pixel, end_pixel=elbow_pixel))
        # 7. Continue washing from elbow to shoulder
        trajectory.add_waypoint(Waypoint(shoulder_point, rotation=None, velocity=0.04, force=[1.0, 1.0], gripper=gripper_value, start_pixel=elbow_pixel, end_pixel=shoulder_pixel))
        # 8. Lift off and return
        trajectory.add_waypoint(Waypoint(shoulder_up, rotation=None, velocity=0.03, force=[], gripper=gripper_value))
        trajectory.add_waypoint(Waypoint(start_up, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
        trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.04, force=[], gripper=gripper_value))
        
    return trajectory