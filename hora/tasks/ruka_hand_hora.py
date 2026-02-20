# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# Modified for 16-actuated + 4-mimic joint configuration
# --------------------------------------------------------

import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from glob import glob
from hora.utils.misc import tprint
from .base.vec_task import VecTask

class RukaHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        self._setup_domain_rand_config(config['env']['randomization'])
        self._setup_priv_option_config(config['env']['privInfo'])
        self._setup_object_info(config['env']['object'])
        self._setup_reward_config(config['env']['reward'])
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.evaluate = self.config['on_evaluation']
        self.priv_info_dict = {
            'obj_position': (0, 3), 'obj_scale': (3, 4), 'obj_mass': (4, 5), 'obj_friction': (5, 6), 'obj_com': (6, 9),
        }

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        # --- JOINT MAPPING SETUP ---
        # We define which of the 20 joints are the 16 the AI actually controls
        # Indices: 0,1,3,4,5,7,8,9,11,12,13,15,16,17,18,19 (Excludes 2, 6, 10, 14)
        self.dip_indices = torch.tensor([2, 6, 10, 14], device=self.device, dtype=torch.long)
        self.pip_indices = torch.tensor([1, 5, 9, 13], device=self.device, dtype=torch.long)
        self.active_dof_indices = torch.tensor([i for i in range(self.num_allegro_hand_dofs) if i not in [2, 6, 10, 14]], device=self.device, dtype=torch.long)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.rigid_body_states.shape[1], 3), dtype=torch.float, device=self.device)

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')).float().to(self.device)

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.p_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * config['env']['controller']['pgain']
        self.d_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * config['env']['controller']['dgain']

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self._create_object_asset()

        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 0.5
            allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
            allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            # Disable drive for mimic joints to prevent fighting
            if i in [2, 6, 10, 14]:
                allegro_hand_dof_props['stiffness'][i] = 10.0 # High stiffness to follow PIP
            
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()
        self.envs = []
        self.hand_indices = []
        self.object_indices = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            self.hand_indices.append(self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM))

            object_asset = self.object_asset_list[np.random.choice(len(self.object_type_list), p=self.object_type_prob)]
            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_indices.append(self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM))
            self.envs.append(env_ptr)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0: continue
            scale_key = str(self.randomize_scale_list[n_s])
            sampled_pose = self.saved_grasping_states[scale_key][np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))].clone()
            
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 20:] # Object pose starts after 20 joints
            pos = sampled_pose[:, :20]
            self.allegro_hand_dof_pos[s_ids, :] = pos
            self.prev_targets[s_ids, :20] = pos
            self.cur_targets[s_ids, :20] = pos
            self.init_pose_buf[s_ids, :] = pos.clone()

        obj_idx_cv = self.object_indices[env_ids].to(torch.int32)
        hand_idx_cv = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(obj_idx_cv), len(obj_idx_cv))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_idx_cv), len(env_ids))
        self.progress_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()
        # Filter for the 16 active joints for the policy
        active_pos = self.allegro_hand_dof_pos[:, self.active_dof_indices]
        
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = unscale(active_pos, self.allegro_hand_dof_lower_limits[self.active_dof_indices], self.allegro_hand_dof_upper_limits[self.active_dof_indices]).unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, self.active_dof_indices].unsqueeze(1)
        
        cur_obs_combined = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_combined], dim=1)
        
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()
        self.obs_buf[:, :t_buf.shape[1]] = t_buf

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        
        # Apply 16 actions to the 16 active joints
        active_targets = self.prev_targets[:, self.active_dof_indices] + 1 / 24 * self.actions
        self.cur_targets[:, self.active_dof_indices] = tensor_clamp(active_targets, self.allegro_hand_dof_lower_limits[self.active_dof_indices], self.allegro_hand_dof_upper_limits[self.active_dof_indices])
        
        # MIMIC COUPLING: Set DIP targets to match PIP targets
        self.cur_targets[:, self.dip_indices] = self.cur_targets[:, self.pip_indices]
        
        self.prev_targets[:] = self.cur_targets.clone()
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

    def update_low_level_control(self):
        self._refresh_gym()
        # Force the mimicry one last time before stepping physics
        self.cur_targets[:, self.dip_indices] = self.cur_targets[:, self.pip_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]