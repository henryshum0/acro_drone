import numpy as np
import do_mpc
import casadi as ca

from .mpc_cfg import MpcCfg

class MPCControllerWrapper:
	"""A wrapper around the MPCController to handle multiple environments and trajectory updates."""
	
	def __init__(self, cfg: MpcCfg, num_envs: int):
		self.cfg = cfg
		self.num_envs = num_envs
		self.mpc_controllers = [MPCController(cfg) for _ in range(num_envs)]

	def reset_idx(self, env_ids: list, x0: np.ndarray, horizon: np.ndarray):
		self._set_horizon(env_ids, horizon)
		for env_id in env_ids:
			self.mpc_controllers[env_id].reset(x0[env_id], horizon[env_id])

	def _set_horizon(self, env_ids, horizon):
		for env_id in env_ids:
			self.mpc_controllers[env_id]._set_horizon(horizon[env_id])

	def make_step(self, x0, horizon):
		return np.array([mpc.make_step(x0[i], horizon[i]) for i, mpc in enumerate(self.mpc_controllers)])

class MPCController:
	"""Model Predictive Controller for trajectory tracking."""
	
	def __init__(self, cfg: MpcCfg):
		self.cfg = cfg
		self._horizon = None

	def reset(self, x0, horizon):
		self._set_horizon(horizon)
		self._setup(x0)

	def _set_horizon(self, horizon):
		self._horizon = horizon
		if self._horizon is None:
			raise ValueError("MPC horizon is None.")
		if len(self._horizon) != self.cfg.horizon + 1:
			raise ValueError("MPC horizon length does not match configuration.")
		if len(self._horizon[0]) != 13:
			raise ValueError("Each horizon step should have 13 elements (pos(3), quat(4), vel(3), body_rate(3)).")
	
	def make_step(self, x0, horizon):
		self._set_horizon(horizon)
		self._current_x = x0
		u0 = self.mpc.make_step(x0)
		return u0
	
	def _setup(self, x0):
		self._current_x = x0
		model = do_mpc.model.Model("continuous")
		p = model.set_variable(var_type="_x", var_name="p", shape=(3,1))
		quat = model.set_variable(var_type="_x", var_name="quat", shape=(4,1))
		v = model.set_variable(var_type="_x", var_name="v", shape=(3,1))
		body_rate = model.set_variable(var_type="_x", var_name="body_rate", shape=(3,1))

		p_ref = model.set_variable(var_type="_tvp", var_name="p_ref", shape=(3,1))
		quat_ref = model.set_variable(var_type="_tvp", var_name="quat_ref", shape=(4,1))
		v_ref = model.set_variable(var_type="_tvp", var_name="v_ref", shape=(3,1))
		body_rate_ref = model.set_variable(var_type="_tvp", var_name="body_rate_ref", shape=(3,1))

		thrust = model.set_variable(var_type="_u", var_name="thrust", shape=(1,1))
		w = model.set_variable(var_type="_u", var_name="w", shape=(3,1))
		
		g = np.array([0, 0, -self.cfg.gravity])

		model.set_rhs("body_rate", w)
		model.set_rhs("quat", quat_derivative_ca(quat, body_rate))
		model.set_rhs("v", g + quat_apply_ca(quat, ca.vertcat(0, 0, thrust)))
		model.set_rhs("p", v)
		

		model.setup()
		self.mpc = do_mpc.controller.MPC(model)
		params = {
			"n_horizon": self.cfg.horizon,
			"t_step": self.cfg.dt,
			"n_robust": 0,
			"store_full_solution": True,
			
			"nlpsol_opts": {
				'ipopt.linear_solver': 'mumps',
				'ipopt.max_iter': self.cfg.solver_max_iter,
				'ipopt.print_level': 0,
				'ipopt.tol': self.cfg.solver_tol,
			}
		}
		self.mpc.set_param(**params)
		self.mpc.settings.supress_ipopt_output()
		self.tvp_template = self.mpc.get_tvp_template()


		# set up cost function
		m_term = ca.norm_2(p - p_ref)**2 * self.cfg.w_pos + \
			ca.norm_2(quat - quat_ref)**2 * self.cfg.w_quat + \
			ca.norm_2(v - v_ref)**2 * self.cfg.w_vel + \
			ca.norm_2(body_rate - body_rate_ref)**2 * self.cfg.w_body_rate
		
		l_term = ca.norm_2(p - p_ref)**2 * self.cfg.w_pos + \
			ca.norm_2(quat - quat_ref)**2 * self.cfg.w_quat + \
			ca.norm_2(v - v_ref)**2 * self.cfg.w_vel + \
			ca.norm_2(body_rate - body_rate_ref)**2 * self.cfg.w_body_rate + \
			ca.norm_2(w)**2 * self.cfg.w_output

		self.mpc.set_objective(mterm=m_term, lterm=l_term)
		self.mpc.set_rterm(thrust=self.cfg.w_output_derivative, w=self.cfg.w_output_derivative)

		self.mpc.bounds["upper", "_u", "w"] = np.array([self.cfg.max_roll_pitch_rate]*2 + [self.cfg.max_yaw_rate])
		self.mpc.bounds["lower", "_u", "w"] = -np.array([self.cfg.max_roll_pitch_rate]*2 + [self.cfg.max_yaw_rate])
		self.mpc.bounds["upper", "_u", "thrust"] = self.cfg.max_normalized_thrust
		self.mpc.bounds["lower", "_u", "thrust"] = 0


		self.mpc.set_tvp_fun(self._tvp_func)
		self.mpc.setup()
		
		self.mpc.x0 = x0
		self.mpc.set_initial_guess()

	def _tvp_func(self, t_now):
		if self._horizon is None:
			raise RuntimeError("MPC horizon is not set before TVP update.")

    # Get current actual state from the mpc object for the first comparison
    # self.mpc.x0 is a structured array; we need the 'quat' values
		# last_q = self._current_x[3:7].flatten()

		for k in range(self.cfg.horizon + 1):
			ref_data = self._horizon[k]
		# 	target_q = np.array(ref_data[3:7])
			
		# 	# Hemisphere check: if the dot product is negative, q and -q 
		# 	# are 360 degrees apart. Flip to take the "short way".
		# 	if np.dot(target_q, last_q) < 0:
		# 		target_q = -target_q
			
		# 	# Store for the next iteration's comparison
		# 	last_q = target_q

			self.tvp_template['_tvp', k, 'p_ref'] = ref_data[0:3] # Assuming ref_data is [pos, quat, vel, body_rate]
			self.tvp_template['_tvp', k, 'quat_ref'] = ref_data[3:7] # Use the (potentially flipped) q
			self.tvp_template['_tvp', k, 'v_ref'] = ref_data[7:10]
			self.tvp_template['_tvp', k, 'body_rate_ref'] = ref_data[10:13]
			
		return self.tvp_template
		


def quat_apply_ca(quaternion, vector):
	"""Apply rotation represented by quaternion q to vector v using CasADi."""
	q_vec = quaternion[:3]
	q_w = quaternion[3]
	t = 2 * ca.cross(q_vec, vector)
	return vector + q_w * t + ca.cross(q_vec, t)

def quat_derivative_ca(quaternion, angular_v):
    # Unpack using slicing instead of individual indices
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w_q = quaternion[3]

    # Similarly for w (body rates)
    p_rate = angular_v[0]
    q_rate = angular_v[1]
    r_rate = angular_v[2]
    
    return 0.5 * ca.vertcat(
        w_q * p_rate + y * r_rate - z * q_rate,
        w_q * q_rate + z * p_rate - x * r_rate,
        w_q * r_rate + x * q_rate - y * p_rate,
        -x * p_rate - y * q_rate - z * r_rate
    )


def get_quat_error(quaternion, quaternion_ref):
    # Use slicing to avoid "keyword" ambiguity
    x_r, y_r, z_r, w_r = quaternion_ref[0], quaternion_ref[1], quaternion_ref[2], quaternion_ref[3]
    q_ref_inv = ca.vertcat(-x_r, -y_r, -z_r, w_r)

    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    q_err = ca.vertcat(
        w_r*x + (-x_r)*w + (-y_r)*z - (-z_r)*y,
        w_r*y - (-x_r)*z + (-y_r)*w + (-z_r)*x,
        w_r*z + (-x_r)*y - (-y_r)*x + (-z_r)*w,
        w_r*w - (-x_r)*x - (-y_r)*y - (-z_r)*z,
    )
    return q_err