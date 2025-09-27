import ikpy.chain
import numpy as np
import os
from pathlib import Path

"""Class for solving kinematics of Stretch with full 6 DoF FK and IK"""
class StretchKinematicsSolver(object):
    def __init__(self, goal_link="gripper"):
        self.goal_link = goal_link
        repo_dir = Path(__file__).parent.parent.parent
        if goal_link == "camera":
            urdf = os.path.join(repo_dir, "resources", "stretch_camera_color.urdf")
            self.num_joints = 9
            self.active_indices = [3, 4]
            base_elements = None
        elif goal_link == "shaver":
            urdf = os.path.join(repo_dir, "resources", "stretch_ik_shaver.urdf")
            self.num_joints = 13
            self.active_indices = [1, 3, 5, 6, 8, 9]
            base_elements = ["virtual_base_translate"]
        elif goal_link == "spoon":
            urdf = os.path.join(repo_dir, "resources", "stretch_ik_spoon.urdf")
            self.num_joints = 13
            self.active_indices = [1, 3, 5, 6, 8, 9]
            base_elements = ["virtual_base_translate"]
        else:
            if goal_link != "gripper": 
                print("Unknown goal link! Using gripper by default.")
                self.goal_link = "gripper"
            urdf = os.path.join(repo_dir, "resources", "stretch_mobile_base.urdf")
            self.num_joints = 12
            self.active_indices = [1, 3, 5, 6, 8, 9]
            base_elements = ["virtual_base_translate"]
        active_links_mask = np.zeros((self.num_joints,))
        active_links_mask[self.active_indices] = 1
        self.chain = ikpy.chain.Chain.from_urdf_file(urdf, active_links_mask=np.copy(active_links_mask), base_elements=base_elements)
        self.regularization = 0.1
    
    def forward_kinematics(self, js):
        """js needs to be a the same length as the number of active joints specified in the constructor.
        If this is for end-effector kinematics, the order should be:
            1) base translation, 2) lift joint, 3) arm joint, 4) wrist yaw joint, 5) wrist pitch joint, 6) wrist roll joint.
        If this is for camera kinematics, the order should be:
            1) head pan, 2) head tilt. """
        assert len(js) == len(self.active_indices)
        q = self.chain.active_to_full(js, np.zeros(self.num_joints))
        return self.chain.forward_kinematics(q)
    
    def inverse_kinematics_orientation_axis(self, position, direction, j_curr, axis):
        """transform needs to be a 4x4 matrix of the desired end effector pose.
           j_curr needs to be the same as the specification in FK."""
        q_init = self.chain.active_to_full(j_curr, np.zeros(self.num_joints))
        q_soln = self.chain.inverse_kinematics(target_position=position, target_orientation=direction, orientation_mode=axis, initial_position=q_init, regularization_parameter=self.regularization)
        j_soln = self.chain.active_from_full(q_soln)
        return j_soln

    def inverse_kinematics_orientation(self, transform, j_curr):
        """transform needs to be a 4x4 matrix of the desired end effector pose.
           j_curr needs to be the same as the specification in FK."""
        q_init = self.chain.active_to_full(j_curr, np.zeros(self.num_joints))
        q_soln = self.chain.inverse_kinematics_frame(transform, initial_position=q_init, orientation_mode="all")
        j_soln = self.chain.active_from_full(q_soln)
        return j_soln
    
    def inverse_kinematics(self, pos, j_curr):
        """transform needs to be a 3x1 vector of the desired end effector position.
           j_curr needs to be the same as the specification in FK."""
        q_init = self.chain.active_to_full(j_curr, np.zeros(self.num_joints))
        q_soln = self.chain.inverse_kinematics(pos, initial_position=q_init, regularization_parameter=self.regularization)
        j_soln = self.chain.active_from_full(q_soln)
        return j_soln
