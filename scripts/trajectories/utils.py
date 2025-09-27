import copy
import numpy as np

from . import Trajectory

def segment_trajectory(data: dict) -> dict:
    def get_gripper_change_indices(traj: Trajectory) -> list[int]:
        gripper_value = [w.gripper for w in traj]
        gripper_indices = []
        for i in range(1, len(gripper_value)):
            if (gripper_value[i] > 0 and gripper_value[i-1] <= 0) or \
               (gripper_value[i] <= 0 and gripper_value[i-1] > 0):
                gripper_indices.append(i)
        return gripper_indices
    
    def get_force_change_indices(traj: Trajectory) -> list[int]:
        force_value = [w.force for w in traj]
        force_change_indices = []
        for i in range(1, len(force_value)):
            if (len(force_value[i]) == 0 and len(force_value[i-1]) != 0) or \
               (len(force_value[i]) != 0 and len(force_value[i-1]) == 0):
                force_change_indices.append(i)
        return force_change_indices

    def get_pause_indices(traj: Trajectory) -> list[int]:
        pauses = [w.pause for w in traj]
        pause_indices = []
        for i, v in enumerate(pauses):
            if v is not None and v > 0:
                pause_indices.append(i)
        return pause_indices

    def segment_indices(data: dict, indices: list[int], flags: list[bool]) -> dict:
        """ Segment the input trajectory from the data based on the indices.
        Args:
            data (dict): The input trajectory data containing keys like "gripper_value", "segmenting_timestamps", "T_base_gripper", etc.
            indices (list[int]): A list of indices where the segmentation should occur.
            flags (list[bool]): A list of boolean flags indicating whether to repeat the waypoint at the index.
        Returns:
            segmented_data (dict): A copy of the input data, but with the trajectory information segmented.
        For example, if indices = [3, 6] and flags = [True, True], then the waypoints will be separated into the following sections:
        1) waypoints 1-3, 2) waypoints 3-6, 3) waypoints 6-end.
        """
        assert len(indices) == len(flags), "Indices and flags must have the same length."

        segmented_data = copy.deepcopy(data)
        segmented_data["trajectory"] = []
        segmented_data["gripper_pos_pixel"] = []
        
        for i in range(len(indices)+1):
            if i == len(indices):
                ending_index = len(data["trajectory"])
            else:
                index, repeat_flag = indices[i], flags[i]
                ending_index = index
            if i == 0:
                starting_index = 0
            else:
                starting_index = indices[i-1]
                if repeat_flag:
                    starting_index -= 1

            segmented_data["trajectory"].append(data["trajectory"][starting_index:ending_index])
            segmented_data["gripper_pos_pixel"].append(data["gripper_pos_pixel"][starting_index:ending_index])
                
        return segmented_data

    assert isinstance(data["trajectory"], Trajectory), "Data must be unsegmented, with 'trajectory' field being a Trajectory object."
    gripper_change_indices = get_gripper_change_indices(data["trajectory"])
    force_change_indices = get_force_change_indices(data["trajectory"])

    pause_indices = get_pause_indices(data["trajectory"])
    all_indices = [(i, False) for i in gripper_change_indices] + \
                  [(i, True) for i in force_change_indices] + \
                  [(i, True) for i in pause_indices]
    all_indices.sort(key=lambda x: x[0])
    segmented_data = segment_indices(data, [i[0] for i in all_indices], [i[1] for i in all_indices])
    return segmented_data