# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# Modified for 20-DOF Hand Grasp Generation
# --------------------------------------------------------

import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis, quat_mul, tensor_clamp, to_torch
from hora.tasks.ruka_hand_hora import RukaHandHora

class RukaHandGrasp(RukaHandHora):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        # 20 joints + 7 root state (pos/rot) = 27 columns
        self.saved_grasping_states = torch.zeros((0, 27), dtype=torch.float, device=self.device)
        
        # Expanded to 20 joints. Added 0.0 at mimic indices [2, 6, 10, 14]
        # These will be overwritten by the mimic logic anyway.
        self.canonical_pose = [
            -0.17,     # base pitch
            -0.465732, # wrist_yaw
            -0.191799, # index_splay
            1.7,       # index_mcp
            0.0,       # index_pip
            0.0,       # index_dip
            1.7,      # mid_mcp
            0.0,       # mid_pip
            0.0,       # mid_dip
            -0.301426, # ring_splay
            1.7,       # ring_mcp
            0.0,       # ring_pip
            0.0,       # ring_dip
            -0.298132, # pinky_splay
            1.7,       # pinky_mcp
            0.0,       # pinky_pip
            0.0,       # pinky_dip
            0.5,      # thumb_cmc
            -0.77,     # thumb_mcp
            0.0,       # thumb_ip
        ]
        
        # Joint Indices for 20-DOF hand
        self.dip_indices = [5, 8, 12, 16]
        self.pip_indices = [4, 7, 11, 15]
        
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def reset_idx(self, env_ids):
        # ... [Keep randomization of mass/PD gains the same as your provided snippet] ...
        
        # Generate random values for 20 joints
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # Cache successful grasps
        self.rb_forces[env_ids, :, :] = 0.0
        success = self.progress_buf[env_ids] == self.max_episode_length
        
        all_states = torch.cat([
            self.allegro_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)
        
        if success.any():
            self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
            print("success")
            print('current cache size:', self.saved_grasping_states.shape[0])

        if len(self.saved_grasping_states) >= 5e4:
            name = f'cache/{self.grasp_cache_name}_grasp_50k_s{str(self.base_obj_scale).replace(".", "")}.npy'
            np.save(name, self.saved_grasping_states[:50000].cpu().numpy())
            print(f"Saved 50k grasps to {name}")
            exit()

        # Reset object pose
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        # Force upright for canonical start
        new_object_rot[:] = 0
        new_object_rot[:, -1] = 1
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # Reset Hand Position (20 joints)
        pos = to_torch(self.canonical_pose, device=self.device)[None].repeat(len(env_ids), 1)


        # print(f"DEBUG: pos shape: {pos.shape}")
        # print(f"DEBUG: rand_floats slice shape: {rand_floats[:, 5:5 + self.num_allegro_hand_dofs].shape}")
        print("717")
        pos += 0.25 * rand_floats[:, 5:5 + self.num_allegro_hand_dofs]
        
        # MANUALLY ENFORCE MIMICRY for the reset pose
        pos[:, self.dip_indices] = pos[:, self.pip_indices]
        
        pos = tensor_clamp(pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        # ... [Rest of buffers reset] ...

    def compute_reward(self, actions):
        # NOTE: Fingertip indices shift in 20-DOF URDF. 
        # Typically Index=5, Middle=10, Ring=15, Thumb=20 depending on URDF tree.
        # Adjusted indices from [4, 8, 12, 16] to [5, 10, 15, 20]
        
        def list_intersect(li, hash_num):
            obj_id = 21 # Object is the next actor after 20 hand bodies
            query_list = [obj_id * hash_num + 5, obj_id * hash_num + 10, obj_id * hash_num + 15, obj_id * hash_num + 20]
            return len(np.intersect1d(query_list, li))

        assert self.device == 'cpu'
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        contact_list = [list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000) for contact in contacts]
        contact_condition = to_torch(contact_list, device=self.device)

        obj_pos = self.rigid_body_states[:, [-1], :3]
        # Fingertip rigid body indices
        finger_pos = self.rigid_body_states[:, [6, 10, 15, 20, 24], :3]
        
        cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < 0.1).all(-1)
        cond2 = contact_condition >= 2
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        
        cond = cond1.float() * cond2.float() * cond3.float()
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
