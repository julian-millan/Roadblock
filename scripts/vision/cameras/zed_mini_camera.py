import cv2
import numpy as np
import pyzed.sl as sl

from . import BaseCamera

class ZedMiniCamera(BaseCamera):
    def __init__(self, rotation):
        super().__init__(rotation)
        self.camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception("Cannot open ZED camera")
        
        self._width = 1920
        self._height = 1080

        self.color_frame = sl.Mat()
        self.depth_frame = sl.Mat()
        self.point_cloud = sl.Mat()

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    def get_new_frames(self):
        """ This function returns color image in RGB format."""
        runtime_parameters = sl.RuntimeParameters()
        if self.camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.color_frame, sl.VIEW.LEFT)
            self.camera.retrieve_image(self.depth_frame, sl.VIEW.DEPTH)
            self.camera.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
            color_image = self.color_frame.get_data()[..., :3]
            depth_image = self.depth_frame.get_data()[...,0]
            return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_image
        else:
            return self.get_new_frames()
        
    def get_3d_from_pixel(self, pixel, transform=np.eye(4)):
        err, point_cloud_value = self.point_cloud.get_value(int(pixel[0]), int(pixel[1]))
        if err == sl.ERROR_CODE.SUCCESS:
            point = np.array(point_cloud_value[:3])
        else:
            print(f"Cannot get 3D for pixel ({pixel[0]}, {pixel[1]})")
            point = None
        return None if point is None else transform[:3,:3] @ point + transform[:3,3]
    
    def get_pixel_from_3d(self, point, transform=np.eye(4)):
        depth_intrinsics = self.get_depth_intrinsics()
        transformed_point = np.linalg.inv(transform[:3,:3]) @ (point - transform[:3,3])
        pixel = depth_intrinsics @ (transformed_point / transformed_point[2])
        return pixel[:2].astype(int)
        
    def get_color_intrinsics(self):
        intrinsics =  self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy
        K = np.array([
            [fx,  0, cx],
            [0,  fy, cy],
            [0,   0,  1]
        ])
        return K
    
    def get_depth_intrinsics(self):
        return self.get_color_intrinsics()

    def close(self):
        self.camera.close()

if __name__ == "__main__":
    camera = ZedMiniCamera()
    color_image, depth_image = camera.get_new_frames()
    test_normal = camera.get_normal_from_pixel([100, 100])
    print("Test normal:", test_normal)
    camera.close()