import pyrealsense2 as rs
import numpy as np

from . import BaseCamera

class RealSenseCamera(BaseCamera):
    def __init__(self, serial_number, model, rotation, options=None):
        """ Configure a camera with the given serial number, and also start the stream. """
        super().__init__(rotation)
        devices = rs.context().query_devices()
        camera_found = any(device.get_info(rs.camera_info.serial_number) == serial_number for device in devices)
        if not camera_found:
            raise Exception(f"RealSense camera with serial number {serial_number} not found!")

        self.camera = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        if model == "D435i":
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
            self.color_only = False
            self._width = 1280
            self._height = 720
        elif model == "D405":
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
            self.color_only = True
            self._width = 1280
            self._height = 720
        self.color_frame = None
        if not self.color_only:
            self.depth_frame = None

            self.align = rs.align(rs.stream.color)

            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 5)
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling_filter = rs.hole_filling_filter(2)
            
        profile = self.camera.start(config)
        if options is not None:
            sensor = profile.get_device().query_sensors()[0]
            if "exposure" in options:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, options["exposure"])
            if "gain" in options:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.gain, options["gain"])
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    def get_new_frames(self):
        """ Fetch a set of new frames from the stream, align depth to color, and then
            return first the color frame and then the aligned depth frame. """
        frames = self.camera.wait_for_frames()
        if not self.color_only:
            aligned_frames = self.align.process(frames)
            self.color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not self.color_frame or not depth_frame:
                return self.get_new_frames()
            color_image = np.asanyarray(self.color_frame.get_data())
            processed_frame = self.spatial_filter.process(depth_frame)
            processed_frame = self.temporal_filter.process(processed_frame)
            processed_frame = self.hole_filling_filter.process(processed_frame)
            self.depth_frame = processed_frame.as_depth_frame()
            depth_image = np.asanyarray(self.depth_frame.get_data())
            return color_image, depth_image
        else:
            self.color_frame = frames.get_color_frame()
            if not self.color_frame:
                return self.get_new_frames()
            color_image = np.asanyarray(self.color_frame.get_data())
            return color_image
    
    def get_3d_from_pixel(self, pixel, transform=np.eye(4)):
        """ Traces back to a 3D point given a specific pixel in a depth image frame.
            Usually, this depth_frame should be the one aligned with the color frame.
            Requires the first component of the pixel to be along the horizontal axis (+ means right), 
            and the second component of the pixel to be along the vertical axis (+ means down). """
        depth_intrinsics = self.depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        dist = self.depth_frame.get_distance(pixel[0], pixel[1])
        if dist < 0.001:
            print(f"Cannot get distance for pixel ({pixel[0]}, {pixel[1]})")
            point = None
        else:
            point = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, pixel, dist))
        return None if point is None else transform[:3,:3] @ point + transform[:3,3]
    
    def get_pixel_from_3d(self, point, transform=np.eye(4)):
        depth_intrinsics = self.depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        pixel = np.array(rs.rs2_project_point_to_pixel(depth_intrinsics,
            np.linalg.inv(transform[:3,:3]) @ (point - transform[:3,3])))
        return pixel.astype(int)
    
    def get_intrinsics_from_frame(self, frame):
        intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.ppx
        cy = intrinsics.ppy
        K = np.array([
            [fx,  0, cx],
            [0,  fy, cy],
            [0,   0,  1]
        ])
        return K

    def get_color_intrinsics(self):
        if self.color_frame is None:
            self.get_new_frames()
        return self.get_intrinsics_from_frame(self.color_frame)
    
    def get_depth_intrinsics(self):
        if self.color_only:
            raise Exception("No depth stream defined for the camera!")
        if self.depth_frame is None:
            self.get_new_frames()
        return self.get_intrinsics_from_frame(self.depth_frame)
    
    def close(self):
        self.camera.stop()