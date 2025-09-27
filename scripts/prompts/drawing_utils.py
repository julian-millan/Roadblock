import copy
import cv2
import numpy as np


def draw_axes(image, camera_intrinsics, camera_extrinsics):
    """Draws reference axes on the image"""
    
    image_height, image_width = image.shape[0:2]
    
    zero_transform = np.eye(4)
    unitx_transform = np.copy(zero_transform)
    unitx_transform[0,3] = 1
    unity_transform = np.copy(zero_transform)
    unity_transform[1,3] = 1
    unitz_transform = np.copy(zero_transform)
    unitz_transform[2,3] = 1
    zero_camera = (np.linalg.inv(camera_extrinsics) @ zero_transform)[:3,3]
    x_camera = (np.linalg.inv(camera_extrinsics) @ unitx_transform)[:3,3]
    y_camera = (np.linalg.inv(camera_extrinsics) @ unity_transform)[:3,3]
    z_camera = (np.linalg.inv(camera_extrinsics) @ unitz_transform)[:3,3]
    x_image = (camera_intrinsics @ x_camera / x_camera[2] - camera_intrinsics @ zero_camera / zero_camera[2])[:2]
    y_image = (camera_intrinsics @ y_camera / y_camera[2] - camera_intrinsics @ zero_camera / zero_camera[2])[:2]
    z_image = (camera_intrinsics @ z_camera / z_camera[2] - camera_intrinsics @ zero_camera / zero_camera[2])[:2]

    axes_center = np.array([80, image_height-80])
    axes_length = [20, 40, 20]
    output = np.copy(image)
    cv2.arrowedLine(output, axes_center, np.array(np.round(axes_center + x_image / np.linalg.norm(x_image) * axes_length[0]), dtype=int), color=(255,0,0), thickness=3, tipLength=0.2)
    cv2.arrowedLine(output, axes_center, np.array(np.round(axes_center + y_image / np.linalg.norm(y_image) * axes_length[1]), dtype=int), color=(0,255,0), thickness=3, tipLength=0.2)
    cv2.arrowedLine(output, axes_center, np.array(np.round(axes_center + z_image / np.linalg.norm(z_image) * axes_length[2]), dtype=int), color=(0,0,255), thickness=3, tipLength=0.2)

    return output

def blur_threshold(rgb_image, depth_image, threshold, blur_kernel=51):
    """Gaussian blurring on an image, based on a depth threshold"""
    
    rgb_height, rgb_width = rgb_image.shape[0:2]
    depth_height, depth_width = depth_image.shape[0:2]
    depth_copy = np.copy(depth_image)
    if not ((rgb_height == depth_height) and (rgb_width == depth_width)):
        depth_copy = cv2.resize(depth_copy, (rgb_width, rgb_height))
    rgb_blurred = cv2.GaussianBlur(rgb_image, (blur_kernel, blur_kernel), 30)
    blur_mask = (depth_copy >= threshold)[:, :, None]
    return np.where(blur_mask, rgb_blurred, rgb_image)

def white_tint(base_image, white_proportion=0.4):
    white_image = np.array(np.ones(base_image.shape)*255, dtype=np.uint8)
    tinted = base_image * (1 - white_proportion) + white_image * white_proportion
    return np.array(tinted, dtype=np.uint8)

def draw_point_with_velocity(image, pixel, velocity, min_vel, max_vel, radius=5):
    color_1 = np.array([0,50,0])
    color_2 = np.array([0,255,0])
    color = (velocity - min_vel) / (max_vel - min_vel) * (color_2-color_1) + color_1
    cv2.circle(image, pixel, radius, color, -1)

def draw_line_with_force(image, start_point, end_point, force_profile, min_force, max_force, line_width=2):
    if max_force == min_force:
        min_force = 0.0
        max_force = 5.0
    force_magnitude = np.array(force_profile)
    num_steps = len(force_magnitude)
    normalized_force_magnitude = force_magnitude - min_force
    normalized_force_magnitude *= 255.0 / (max_force - min_force)
    normalized_force_magnitude = np.clip(normalized_force_magnitude, 0, 255)
    color_along_line = cv2.applyColorMap(np.array(normalized_force_magnitude, dtype=np.uint8), cv2.COLORMAP_COOL)
    x = np.linspace(start_point[0], end_point[0], num_steps, dtype=np.int32)
    y = np.linspace(start_point[1], end_point[1], num_steps, dtype=np.int32)
    for i in range(len(x) - 1):
        color = tuple(map(int, reversed(color_along_line[i, 0, :])))  # This does RGB with reversed()
        cv2.line(image, (x[i], y[i]), (x[i + 1], y[i + 1]), color, line_width)

def overlay_waypoints(base_image, pixel_pos_grouped, velocity_grouped=None, force_grouped=None, circle_size=5, line_width=3, crop=False):
    overlay_color1 = (0,0,255)
    overlay_color2 = (255,0,0)
    image_height, image_width = base_image.shape[:2]
    output = []
    if crop:
        margin = 30
        starting_pos = pixel_pos_grouped[0][0]
    if velocity_grouped is not None:
        velocity_all = [vel for vel_list in velocity_grouped for vel in vel_list]
        min_vel = np.min(velocity_all)
        max_vel = np.max(velocity_all)
    for k in range(len(pixel_pos_grouped)):
        pixel_pos_list = pixel_pos_grouped[k]
        if velocity_grouped is not None:
            vel_list = velocity_grouped[k]
        min_force = None
        max_force = None
        if force_grouped is not None:
            force_list = force_grouped[k]
            nonzero_force_mags = [force for force_section in force_list[1:] for force in force_section if force > 0.1]
            min_force = np.min(nonzero_force_mags) if len(nonzero_force_mags) > 0 else None
            max_force = np.max(nonzero_force_mags) if len(nonzero_force_mags) > 0 else None
        next_image = np.copy(base_image)
        for i in range(1,len(pixel_pos_list)):
            if force_grouped is not None and np.sum(np.abs(force_list[i])) > 0.1:
                draw_line_with_force(next_image, (round(pixel_pos_list[i-1][0]), round(pixel_pos_list[i-1][1])), (round(pixel_pos_list[i][0]), round(pixel_pos_list[i][1])), force_list[i], min_force, max_force, line_width=line_width)
            else:
                draw_line_with_force(next_image, (round(pixel_pos_list[i-1][0]), round(pixel_pos_list[i-1][1])), (round(pixel_pos_list[i][0]), round(pixel_pos_list[i][1])), [0,0], min_force, max_force, line_width=line_width)
            if i != len(pixel_pos_list) - 1:
                draw_point_with_velocity(next_image, (round(pixel_pos_list[i][0]), round(pixel_pos_list[i][1])), vel_list[i], min_vel, max_vel, radius=circle_size)
        cv2.rectangle(next_image, (round(pixel_pos_list[0][0])-circle_size-2, round(pixel_pos_list[0][1])-circle_size-2), (round(pixel_pos_list[0][0])+circle_size+2, round(pixel_pos_list[0][1])+circle_size+2), overlay_color1, -1)
        cv2.rectangle(next_image, (round(pixel_pos_list[-1][0])-circle_size-2, round(pixel_pos_list[-1][1])-circle_size-2), (round(pixel_pos_list[-1][0])+circle_size+2, round(pixel_pos_list[-1][1])+circle_size+2), overlay_color2, -1)
        if not crop:
            output.append(next_image)
        else:
            min_pos = np.min(pixel_pos_list + [starting_pos], axis=0) - margin
            max_pos = np.max(pixel_pos_list + [starting_pos], axis=0) + margin
            min_pos = np.clip(min_pos, [0, 0], [image_width, image_height])
            max_pos = np.clip(max_pos, [0, 0], [image_width, image_height])
            output.append(next_image[round(min_pos[1]):round(max_pos[1]), round(min_pos[0]):round(max_pos[0]), :])
    return copy.deepcopy(output)