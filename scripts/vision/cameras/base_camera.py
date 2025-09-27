from abc import ABC, abstractmethod
import cv2
from itertools import product
import numpy as np
from sklearn.decomposition import PCA

class BaseCamera(ABC):
    """ Base class for camera wrappers.
    If implementing your own camera wrapper, you should inherit from this class and implement the following methods:
    - get_new_frames
    - get_3d_from_pixel
    - get_color_intrinsics
    - close
    You should also define the properties width and height, corresponding to the width and height of each incoming image. 
    """
    def __init__(self, rotation):
        self.rotation = rotation

    @property
    @abstractmethod
    def width(self):
        """ The width of each incoming image, usually defined in the constructor. """
        pass

    @property
    @abstractmethod
    def height(self):
        """ The height of each incoming image, usually defined in the constructor. """
        pass

    @abstractmethod
    def get_new_frames(self):
        """ Fetches the latest frames from the camera (color and/or depth), stores them in some class fields internally,
        and returns the color and depth images as numpy arrays. Color image should be in RGB format.
        """
        pass

    @abstractmethod
    def get_3d_from_pixel(self, pixel, transform=np.eye(4)):
        """ Traces back to a 3D point given a specific pixel in a depth image frame. Usually, this depth frame should be the one aligned with the color frame.
        Optionally, you can provide a 4x4 transformation matrix to transform the 3D point to a different coordinate system.
        """
        pass

    @abstractmethod
    def get_pixel_from_3d(self, point, transform=np.eye(4)):
        """ Projects a 3D point to a pixel in the color image frame.
        Optionally, you can provide a 4x4 transformation matrix to transform the 3D point to a different coordinate system before projection.
        """
        pass

    @abstractmethod
    def get_color_intrinsics(self):
        """ Returns the intrinsic matrix of the color (RGB) camera. The intrinsic matrix should be a 3x3 numpy array."""
        pass

    @abstractmethod
    def close(self):
        """ Closes the camera."""
        pass

    def get_depth_intrinsics(self):
        """ Returns the intrinsic matrix of the depth camera. The intrinsic matrix should be a 3x3 numpy array.
        """
        raise NotImplementedError("This function should be implemented in the derived class if it were to be used.")

    def get_normal_from_pixel(self, pixel, transform=np.eye(4)):
        """ Computes the normal of the surface at the given pixel in the depth image.
        The normal is computed by fitting a plane to the 3D points in a window around the pixel.
        The direction of the normal is determined by whichever direction points towards the camera.
        Optionally, you can provide a 4x4 transformation matrix to transform the normal to a different coordinate system.
        """
        half_window = 2
        points_surface = []
        while len(points_surface) < 3:
            half_window += 1
            points_surface = []
            for dx, dy in product(range(-half_window, half_window+1), repeat=2):
                px, py = pixel[0] + dx, pixel[1] + dy
                point = self.get_3d_from_pixel([px, py])
                if point is None:
                    continue
                points_surface.append(point)
        points_surface = np.array(points_surface)
        pca = PCA(n_components=3).fit(points_surface)
        normal = pca.components_[2]
        normal /= np.linalg.norm(normal)
        surface_center = np.mean(points_surface, axis=0)
        if np.linalg.norm(surface_center) > np.linalg.norm(surface_center + normal*0.001):
            return transform[:3,:3] @ normal
        else:
            return transform[:3,:3] @ (-normal)
        
    def depth_map_from_image(self, depth_image):
        """ Converts a depth image to a colored depth map, output in RGB convention."""
        depth_min = depth_image.min()
        depth_max = depth_image.max()        
        depth_norm = (depth_image - depth_min) / (depth_max - depth_min)  # range [0,1]
        depth_image = (depth_norm * 255).astype(np.uint8)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        return depth_image