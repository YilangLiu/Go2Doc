class Flip(go2_base.Go2Env):
  """Flip task for Go2."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.FULL_FLAT_TERRAIN_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._handstand_q = jp.array(self._mj_model.keyframe("handstand").qpos)
    self._footstand_q = jp.array(self._mj_model.keyframe("footstand").qpos)
    self._crouch_q = jp.array(self._mj_model.keyframe("pre_recovery").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self._handstand_pose = jp.array(
        self._mj_model.keyframe("handstand").qpos[7:]
    )
    self._footstand_pose = jp.array(
        self._mj_model.keyframe("footstand").qpos[7:]
    )

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    self._z_des = 0.45
    self._z_des_handstand = 0.45
    self._z_des_footstand = 0.50
    self._z_des_stand = 0.32

    self._desired_base_quat = self._init_q[3:7]
    self._desired_base_quat_handstand = self._mj_model.keyframe("handstand").qpos[3:7]
    self._desired_base_quat_footstand = self._mj_model.keyframe("footstand").qpos[3:7]

    self._desired_forward_vec = jp.array([0, 0, -1])
    self._desired_forward_vec_handstand = jp.array([0, 0, -1])
    self._desired_forward_vec_footstand = jp.array([0, 0, 1])

    self._joint_ids = jp.array([6, 7, 8, 9, 10, 11])
    self._joint_ids_handstand = jp.array([6, 7, 8, 9, 10, 11])
    self._joint_ids_footstand = jp.array([0, 1, 2, 3, 4, 5])
    self._joint_ids_stand = jp.arange(12)

    self._joint_pose = self._default_pose[self._joint_ids]
    self._joint_pose_handstand = self._default_pose[self._joint_ids_handstand]
    self._joint_pose_footstand = self._default_pose[self._joint_ids_footstand]
    self._joint_pose_stand = self._default_pose[self._joint_ids_stand]

    geom_names = [
        "fl_calf_0",
        "fl_calf_1",
        "fr_calf_0",
        "fr_calf_1",
        "fl_thigh_0",
        "fr_thigh_0",
        "fl_hip_0",
        "fr_hip_0",
    ]
    self._unwanted_contact_geom_ids = np.array(
        [self._mj_model.geom(name).id for name in geom_names]
    )
    feet_geom_names = ["RR", "RL"]
    self._feet_geom_ids = np.array(
        [self._mj_model.geom(name).id for name in feet_geom_names]
    )

    geom_names_handstand = [
        "fl_calf_0",
        "fl_calf_1",
        "fr_calf_0",
        "fr_calf_1",
        "fl_thigh_0",
        "fr_thigh_0",
        "fl_hip_0",
        "fr_hip_0",
    ]
    self._unwanted_contact_geom_ids_handstand = np.array(
        [self._mj_model.geom(name).id for name in geom_names_handstand]
    )
    feet_geom_names_handstand = ["RR", "RL"]
    self._feet_geom_ids_handstand = np.array(
        [self._mj_model.geom(name).id for name in feet_geom_names_handstand]
    )

    geom_names_footstand = [
        "rl_calf_0",
        "rl_calf_1",
        "rr_calf_0",
        "rr_calf_1",
        "rl_thigh_0",
        "rr_thigh_0",
        "rl_hip_0",
        "rr_hip_0",
    ]
    self._unwanted_contact_geom_ids_footstand = np.array(
        [self._mj_model.geom(name).id for name in geom_names_footstand]
    )
    feet_geom_names_footstand = ["FR", "FL"]
    self._feet_geom_ids_footstand = np.array(
        [self._mj_model.geom(name).id for name in feet_geom_names_footstand]
    )

    geom_names_stand = [
        "fl_calf_0",
        "fl_calf_1",
        "fr_calf_0",
        "fr_calf_1",
        "fl_thigh_0",
        "fr_thigh_0",
        "fl_hip_0",
        "fr_hip_0",
        "rl_calf_0",
        "rl_calf_1",
        "rr_calf_0",
        "rr_calf_1",
        "rl_thigh_0",
        "rr_thigh_0",
        "rl_hip_0",
        "rr_hip_0",
    ]
    self._unwanted_contact_geom_ids_stand = np.array(
        [self._mj_model.geom(name).id for name in geom_names_stand]
    )
    self._feet_geom_ids_stand = jp.array([])

    self.max_episode_length = self._config.episode_length
    self.flip_timing = int(self.max_episode_length)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, reset_rng = jax.random.split(rng)

    init_from_crouch = jax.random.bernoulli(
        reset_rng, self._config.init_from_crouch
    )

    qpos = jp.where(init_from_crouch, self._crouch_q, self._init_q)
    # qpos = self._init_q
    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    qvel_nonzero = jp.zeros(self.mjx_model.nv)
    rng, key = jax.random.split(rng)
    qvel_nonzero = qvel_nonzero.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )
    qvel = jp.where(init_from_crouch, jp.zeros(self.mjx_model.nv), qvel_nonzero)

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    info = {
        "step": 0,
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
    }
    self.t = 0
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # Initial task is to perform handstand
    contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._unwanted_contact_geom_ids_handstand
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = state.data.ctrl + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    # now we have to decide what stage we are in considering the current step 
    # footstand: 0 -> 100  
    # handstand: 100 -> end
    step = state.info["step"]
    self.t = step

    self._unwanted_contact_geom_ids = jp.where(step < self.flip_timing, self._unwanted_contact_geom_ids_footstand, self._unwanted_contact_geom_ids_handstand)
    self._feet_geom_ids = jp.where(step < self.flip_timing, self._feet_geom_ids_footstand, self._feet_geom_ids_handstand)
    self._z_des = jp.where(step < self.flip_timing, self._z_des_footstand, self._z_des_handstand)
    self._joint_ids = jp.where(step < self.flip_timing, self._joint_ids_footstand, self._joint_ids_handstand)
    self._joint_pose = jp.where(step < self.flip_timing, self._joint_pose_footstand, self._joint_pose_handstand)
    self._desired_forward_vec = jp.where(step < self.flip_timing, self._desired_forward_vec_footstand, self._desired_forward_vec_handstand)
    self._desired_base_quat = jp.where(step < self.flip_timing, self._desired_base_quat_footstand, self._desired_base_quat_handstand)

    contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._unwanted_contact_geom_ids
    ])
    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data, state.info, contact)

    rewards = self._get_reward(data, action, state.info, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["step"] += 1
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> jax.Array:
    del info  # Unused.
    fall_termination = self.get_upvector(data)[-1] < -0.25 
    contact_termination = jp.any(contact)
    energy = jp.sum(jp.abs(data.actuator_force) * jp.abs(data.qvel[6:]))
    energy_termination = energy > self._config.energy_termination_threshold
    return fall_termination | contact_termination | energy_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> Dict[str, jax.Array]:
    del contact  # Unused.

    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )
    phase = jp.pi * self.t / self.max_episode_length
    state = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        noisy_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        jp.sin(phase),
        jp.cos(phase),
        jp.sin(phase / 2),
        jp.cos(phase / 2),
        jp.sin(phase / 4),
        jp.cos(phase / 4),
    ])

    accelerometer = self.get_accelerometer(data)
    linvel = self.get_local_linvel(data)
    angvel = self.get_global_angvel(data)
    torso_height = data.site_xpos[self._imu_site_id][2]

    privileged_state = jp.hstack([
        state,
        gyro,
        accelerometer,
        linvel,
        angvel,
        joint_angles,
        joint_vel,
        data.actuator_force,
        torso_height,
        jp.sin(phase),
        jp.cos(phase),
        jp.sin(phase / 2),
        jp.cos(phase / 2),
        jp.sin(phase / 4),
        jp.cos(phase / 4),
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    forward = data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
    joint_torques = data.actuator_force
    torso_height = data.site_xpos[self._imu_site_id][2]

    return {
        "height": self._reward_height(torso_height),
        "orientation": self._cost_orientation(data),
        # "orientation": self._reward_orientation(
        #     forward, self._desired_forward_vec
        # ),
        # "rest_orientation": self._cost_orientation(self.get_upvector(data)),
        # "stand_still": self._cost_stand_still(data.qpos[7:]),
        "contact": self._cost_contact(data),
        "action_rate": self._cost_action_rate(action, info),
        "torques": self._cost_torques(joint_torques),
        "termination": done,
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "pose": self._cost_pose(data.qpos[7:]),
        "stay_still": self._cost_stay_still(data.qvel[:6]),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
    }

  def _cost_stay_still(self, qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qvel[:2])) + jp.square(qvel[5])

  # def _reward_orientation(
      # self, forward_vec: jax.Array, up_vec: jax.Array
  # ) -> jax.Array:
    # cos_dist = jp.dot(forward_vec, up_vec)
    # normalized = 0.5 * cos_dist + 0.5
    # return jp.square(normalized) 
  
  def _cost_orientation(self, data: mjx.Data) -> jax.Array:
    curr_quat = data.qpos[3:7]
    quat_error = math.quat_sub(curr_quat, self._desired_base_quat)
    return jp.sum(jp.square(quat_error)) 

  def _reward_height(self, torso_height: jax.Array) -> jax.Array:
    height = jp.min(jp.array([torso_height, self._z_des]))
    error = self._z_des - height
    return jp.exp(-error / 1.0)

  def _cost_contact(self, data: mjx.Data) -> jax.Array:
    feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_ids
    ])
    return jp.any(feet_contact)

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos[self._joint_ids] - self._joint_pose))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    return jp.sum(jp.square(act - info["last_act"]))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))