import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch  # Still using torch for the problem definition
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # Removed 3d import
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import time

# Import the problem definition from the CPL code's problem.py
# Make sure problem.py is in the same directory or accessible
from problem import get_problem  # , OptimizationProblem # Removed OptimizationProblem


class HomotopyEnv(gym.Env):
    """
    Custom Gym environment for Homotopy Optimization.

    State: [x1, x2, ..., xd, t]
    Action: [dx1, dx2, ..., dxd] (update step for x)
    Reward: Based on decrease in H(x, t) and reaching t=1.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, problem_name='ackley', max_steps=500, t_increase_rate=0.005, action_scale=0.1):
        super().__init__()

        self.problem = get_problem(problem_name)  # Changed from: self.problem: OptimizationProblem = get_problem(problem_name)
        self.n_dim = self.problem.n_dim
        self._max_steps = max_steps
        self._t_increase_rate = t_increase_rate  # How much t increases per step
        self._action_scale = action_scale  # Scales the agent's output action

        # Define action space: Continuous vector for delta_x
        # Values between -1 and 1, will be scaled by _action_scale
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_dim,), dtype=np.float32)

        # Define observation space: [x1, ..., xd, t]
        # Define reasonable bounds for x based on typical problem ranges
        low_bounds = np.full(self.n_dim + 1, -10.0, dtype=np.float32)
        high_bounds = np.full(self.n_dim + 1, 10.0, dtype=np.float32)
        low_bounds[-1] = 0.0  # Lower bound for t
        high_bounds[-1] = 1.0  # Upper bound for t
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        self.current_step = 0
        self.state = None
        self.current_h_value = None
        self.fig = None
        self.ax = None
        self.surface = None

    def _get_obs(self):
        return self.state.astype(np.float32)

    def _get_info(self):
        # Return current x and H(x, t) value
        return {"x": self.state[:-1], "h_value": self.current_h_value, "t": self.state[-1]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Initial state: Random x within bounds, t starts near 0
        initial_x = self.np_random.uniform(low=-5.0, high=5.0, size=self.n_dim)
        initial_t = 0.01  # Start slightly above 0

        self.state = np.concatenate([initial_x, [initial_t]]).astype(np.float32)

        # Calculate initial H value
        x_tensor = torch.tensor(initial_x, dtype=torch.float32).unsqueeze(0)
        t_tensor = torch.tensor([[initial_t]], dtype=torch.float32)
        with torch.no_grad():
            self.current_h_value, _ = self.problem.evaluate(x_tensor, t_tensor)
            self.current_h_value = self.current_h_value.item()

        observation = self._get_obs()
        info = self._get_info()

        # print(f"Env Reset: Initial State={observation}, Initial H={self.current_h_value:.4f}") # Debug
        return observation, info

    def step(self, action):
        # --- Apply Action ---
        # Scale action and update x part of the state
        delta_x = action * self._action_scale
        current_x = self.state[:-1]
        new_x = current_x + delta_x

        # --- Update Homotopy Parameter t ---
        current_t = self.state[-1]
        new_t = min(1.0, current_t + self._t_increase_rate)  # Increase t, cap at 1.0

        # --- Update State ---
        # Clip x to stay within reasonable bounds (optional, but can help stability)
        new_x = np.clip(new_x, -10.0, 10.0)
        self.state = np.concatenate([new_x, [new_t]]).astype(np.float32)

        # --- Calculate New H Value ---
        x_tensor = torch.tensor(new_x, dtype=torch.float32).unsqueeze(0)
        t_tensor = torch.tensor([[new_t]], dtype=torch.float32)
        with torch.no_grad():
            new_h_value, _ = self.problem.evaluate(x_tensor, t_tensor)
            new_h_value = new_h_value.item()

        # --- Calculate Reward ---
        # Reward based on reduction in H value
        h_value_decrease = self.current_h_value - new_h_value
        reward = float(h_value_decrease * 10.0)  # Scale reward

        # Bonus reward for reaching t=1 with low H value
        if new_t >= 1.0 and new_h_value < 0.1:  # Threshold for 'good' solution
            reward += 50.0
        elif new_t >= 1.0:
            reward += 10.0  # Smaller bonus just for reaching t=1

        # Small penalty per step to encourage efficiency
        reward -= 0.01

        # Update current H value for next step
        self.current_h_value = new_h_value

        # --- Check Termination Conditions ---
        self.current_step += 1
        terminated = bool(new_t >= 1.0)  # End when t reaches 1
        truncated = bool(self.current_step >= self._max_steps)  # End if max steps reached

        # --- Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info()

        # if self.current_step % 50 == 0: # Debug print
        #     print(f"Step {self.current_step}: Action={action}, State={observation}, Reward={reward:.4f}, H={new_h_value:.4f}, Term={terminated}, Trunc={truncated}")

        return observation, reward, terminated, truncated, info

    def render(self):
        # Basic rendering (optional)
        if self.state is not None:
            print(f"Step: {self.current_step}, t: {self.state[-1]:.3f}, x: {self.state[:-1]}, H: {self.current_h_value:.4f}")

    def close(self):
        pass

    def render_3d(self, x_history, h_history):
        """
        Renders the 2D plot of the function and the agent's trajectory.
        """
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_tensor = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)
                t_tensor = torch.tensor([[1.0]], dtype=torch.float32)  # Visualize at t=1
                with torch.no_grad():
                    Z[i, j], _ = self.problem.evaluate(x_tensor, t_tensor)
        # Z = Z.numpy()

        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)  # Changed to 2D subplot
            self.surface = self.ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.5) # Changed to contour plot
            self.ax.set_xlabel('x1')
            self.ax.set_ylabel('x2')
            # self.ax.set_zlabel('H(x, 1)') # Removed z label
            self.ax.set_title(f'Homotopy Function: {self.problem.__class__.__name__}')

        # Plot the agent's trajectory
        x_traj = np.array(x_history)
        h_traj = np.array(h_history)

        if hasattr(self, '_traj_points'):
            for point in self._traj_points:
                point.remove()


        if x_traj.size > 0:
            # Ensure x_traj has shape (n, 2) even for the first point
            if x_traj.ndim == 1:
                x_traj = x_traj.reshape(1, -1)
            self._traj_points = self.ax.plot(x_traj[:, 0], x_traj[:, 1], c='red', marker='o', markersize=3)  # Changed to 2D plot
        else:
            self._traj_points = self.ax.plot([], [], c='red', marker='o', markersize=3)  # Changed to 2D plot

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)  # Add a small delay for better visualization



# --- Main RL Training Script ---
if __name__ == "__main__":
    problem_to_solve = 'ackley'  # Or 'himmelblau', 'rosenbrock'
    print(f"--- Training RL Agent for {problem_to_solve} using Homotopy Environment ---")

    # Create and wrap the environment
    env = HomotopyEnv(problem_name=problem_to_solve, max_steps=500, t_increase_rate=1.0 / 500, action_scale=0.1)

    # It's recommended to wrap the environment in a DummyVecEnv for SB3
    env = DummyVecEnv([lambda: env])

    # Instantiate the agent (PPO is a good default for continuous spaces)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log="./ppo_homotopy_tensorboard/",
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5)

    # Train the agent
    total_timesteps = 10  # Adjust as needed
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("Training finished.")

    # Save the agent
    model_path = f"ppo_{problem_to_solve}_homotopy"
    model.save(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_with_timestamp = f"{model_path}_{timestamp}"
    model.save(model_path_with_timestamp)
    print(f"Model saved to {model_path_with_timestamp}.zip")

    # --- Evaluate the trained agent ---
    print("\n--- Evaluating Trained Agent ---")
    # Load the trained agent (optional, could just use the 'model' variable)
    # model = PPO.load(model_path, env=env)

    obs = env.reset()
    n_eval_episodes = 10
    total_rewards = 0
    final_h_values = []
    final_x_values = []

    # Store data for plotting
    episode_data = []
    h_value_history = []  # Track H values to check for convergence
    convergence_threshold = 1e-5  # Define a threshold for minimal change in H

    print("Starting evaluation...")
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        last_info = {}
        x_history = []  # Store x values for each episode
        h_history = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)  # Corrected unpacking
            truncated = False  # You were assigning truncated but not using the return value
            done = terminated or truncated
            episode_reward += reward
            last_info = info[0]  # Info from the underlying env
            step += 1
            x_history.append(last_info.get('x'))
            h_history.append(last_info.get('h_value'))
            # Access the underlying environment to call render_3d
            env.envs[0].render_3d(x_history, h_history)  # Pass history for plotting

        total_rewards += episode_reward
        final_h_values.append(last_info.get('h_value', np.nan))
        final_x_values.append(last_info.get('x', np.array([np.nan] * env.envs[0].n_dim)))  # Access n_dim from underlying env
        # Ensure 'h_value' is a scalar
        final_h_value = last_info.get('h_value', np.nan)
        if isinstance(final_h_value, np.ndarray):
            final_h_value = final_h_value.item()  # Convert to a scalar if it's an array

        # Convert 'x' array to a string for printing, handle potential errors in conversion
        final_x = last_info.get('x', np.array([np.nan] * env.envs[0].n_dim))
        if isinstance(final_x, np.ndarray):
            final_x_str = np.array2string(final_x, precision=4, separator=', ', floatmode='fixed')
        else:
            final_x_str = str(final_x)

        episode_data.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'final_h': final_h_value,
            'final_x': final_x,
            'steps': step,
            'x_history': x_history,  # Store the x values
            'h_history': h_history
        })
        # Removed the print line causing the error
        # print(f"Eval Episode {episode+1}: Reward={episode_reward:.2f}, Final H={final_h_value:.4f}, Steps={step}")
        h_value_history.append(final_h_value)

        # Check for convergence (stop if H value change is small)
        if len(h_value_history) > 1:
            h_change = np.abs(h_value_history[-1] - h_value_history[-2])
            if h_change < convergence_threshold:
                print(f"Converged after {episode + 1} episodes.  Final H value: {final_h_value:.6f}")
                break  # Exit the evaluation loop

    mean_reward = total_rewards / (episode + 1) # Use the correct number of episodes
    mean_final_h = np.nanmean(final_h_values)
    std_final_h = np.nanstd(final_h_values)

    print("\n--- Evaluation Summary ---")
    print(f"Mean Reward over {episode + 1} episodes: {float(mean_reward):.2f}")
    print(f"Mean Final H(x, t=1) Value: {float(mean_final_h):.6f}")
    print(f"Std Dev Final H(x, t=1) Value: {float(std_final_h):.6f}")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episode + 2), [data['final_h'] for data in episode_data], marker='o') # Use episode+1 for correct x axis
    plt.xlabel('Episode')
    plt.ylabel('Final H(x, t=1)')
    plt.title('Final Homotopy Value vs Episode')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, episode + 2), [data['reward'] for data in episode_data], marker='o') # Use episode+1 for correct x axis
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward vs Episode')
    plt.tight_layout()
    plt.show()

    # Example of plotting the x trajectory for the last episode.
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(episode_data[-1]['x_history']))
    plt.xlabel('Step')
    plt.ylabel('x value')
    plt.title('X trajectory for the last episode')
    plt.tight_layout()
    plt.show()

    # Example of plotting the H trajectory for the last episode.
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(episode_data[-1]['h_history']))
    plt.xlabel('Step')
    plt.ylabel('H value')
    plt.title('H trajectory for the last episode')
    plt.tight_layout()
    plt.show()
