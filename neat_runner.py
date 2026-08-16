import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
import matplotlib.pyplot as plt
import traceback
from collections import deque # Use deque for efficient replay memory

# --- Configuration ---
# IMPORTANT: SET THIS PATH MANUALLY to match Godot's user://AI_data location
AI_DATA_FOLDER = "C:/Users/LOQ/Documents/AI RUNNER/AI_data" # EXAMPLE PATH - CHANGE THIS!

# --- Constants ---
# Model Parameters
INPUT_SIZE = 356  # Vision Grid (22x16=352) + Velocity (2) + Position (2)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4   # Actions: Left, Right, Jump, Idle

# File Paths
MODEL_PATH = "model.pth" # Saves in the script's directory
GAME_STATE_PATH = os.path.join(AI_DATA_FOLDER, "game_state.json")
AI_ACTION_PATH = os.path.join(AI_DATA_FOLDER, "ai_action.json")
REWARD_LOG_PATH = "reward_log.json" # Saves in the script's directory

# RL Hyperparameters
GAMMA = 0.95                # Discount factor for future rewards
LEARNING_RATE = 0.0005      # Optimizer learning rate
MEMORY_SIZE = 10000         # Max size of replay memory (using deque)
BATCH_SIZE = 64             # Number of samples per training batch
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.998       # Rate at which epsilon decreases
EPSILON_MIN = 0.01          # Minimum exploration rate
MAX_STEPS_PER_EPISODE = 3000 # Force episode end after this many steps
SAVE_EVERY_EPISODES = 20    # How often to save model and plot graph
SYNC_DELAY_SECONDS = 0.06   # Delay to help sync with Godot (tune if needed)

# Action Mapping
ACTION_MAP = {0: "left", 1: "right", 2: "jump", 3: "idle"}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

# --- Neural Network Definition ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc_out(x)
        return x

# --- Helper Functions ---

def preprocess_vision_grid(vision_grid):
    """Flattens and normalizes the vision grid."""
    processed_grid = []
    expected_rows, expected_cols = 22, 16
    if not isinstance(vision_grid, list) or not vision_grid or not isinstance(vision_grid[0], list):
        print(f"[ERROR] Invalid vision_grid received: {vision_grid}. Returning zeros.")
        return [0.0] * (expected_rows * expected_cols)
    current_rows = len(vision_grid)
    current_cols = len(vision_grid[0]) if current_rows > 0 else 0
    for r in range(expected_rows):
        for c in range(expected_cols):
            if r < current_rows and c < current_cols:
                val = vision_grid[r][c]
                processed_grid.append({0: 0.0, 1: -1.0, 2: 1.0, 3: 0.5}.get(val, 0.0))
            else:
                processed_grid.append(0.0)
    if len(processed_grid) != expected_rows * expected_cols:
         print(f"[WARN] Processed grid size mismatch: got {len(processed_grid)}, expected {expected_rows * expected_cols}. Trimming/Padding.")
         processed_grid = processed_grid[:expected_rows * expected_cols]
         while len(processed_grid) < expected_rows * expected_cols: processed_grid.append(0.0)
    return processed_grid

def load_model_state(model, path):
    """Loads model state dict if file exists."""
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path))
            print(f"[INFO] Model state loaded from {path}")
        except Exception as e: print(f"[ERROR] Failed to load model state from {path}: {e}")
    else: print(f"[INFO] No existing model found at {path}. Starting fresh.")

def save_model_state(model, path):
    """Saves model state dict."""
    try: torch.save(model.state_dict(), path)
    except Exception as e: print(f"[ERROR] Failed to save model state to {path}: {e}")

# --- MODIFIED read_game_state (Removed initial os.path.exists check) ---
def read_game_state(max_retries=5, retry_delay=0.025):
    """Reads game state from JSON file, handling errors and retrying on empty file/OS errors."""
    # Initial os.path.exists check removed to avoid noise with atomic writes
    last_exception = None # Store last exception for reporting
    for attempt in range(max_retries):
        try:
            # Try to open the file directly
            with open(GAME_STATE_PATH, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if not content:
                    error_type = "empty file"
                    # Fall through to the retry logic below
                else:
                    # If content is not empty, try parsing
                    return json.loads(content) # SUCCESS!

        # Catch specific OS errors that indicate timing/sync issues or file truly missing
        except FileNotFoundError as fnf_err:
             error_type = type(fnf_err).__name__
             last_exception = fnf_err
             # Fall through to the retry logic below
        except PermissionError as perm_err:
             error_type = type(perm_err).__name__
             last_exception = perm_err
             # Fall through to the retry logic below

        except json.JSONDecodeError as e:
            # This is a real error, file exists but content is bad. Don't retry.
            print(f"[ERROR] JSON Decode Error reading game state (Attempt {attempt+1}/{max_retries}): {e}. Content: '{content[:100]}...'")
            last_exception = e
            break # Exit retry loop

        except Exception as e:
            # Catch other unexpected errors. Don't retry these.
            print(f"[ERROR] Unexpected error reading game state (Attempt {attempt+1}/{max_retries}): {e}")
            last_exception = e
            break # Exit retry loop

        # --- Retry Logic (if we fell through due to empty file or recoverable OS error) ---
        if attempt < max_retries - 1:
            # print(f"[WARN] Read failed ({error_type}) on attempt {attempt+1}. Retrying...") # Can be spammy
            time.sleep(retry_delay * (attempt + 1)) # Wait longer each retry
            continue # Go to next attempt
        else:
            # Log final error only after all retries fail for recoverable errors
            print(f"[ERROR] Read failed after {max_retries} attempts ({error_type}). Last error: {last_exception}. Possible sync issue or Godot stopped.")
            return None # Failed all retries

    # If loop finished or broke due to non-recoverable errors
    return None
# --- End of MODIFIED read_game_state ---


def save_action(action_str):
    """Saves the chosen action string to JSON file."""
    try:
        with open(AI_ACTION_PATH, "w", encoding="utf-8") as file:
            json.dump({"action": action_str}, file)
    except Exception as e: print(f"[ERROR] Failed to save action to {AI_ACTION_PATH}: {e}")

def load_reward_log():
    """Loads reward history from JSON file."""
    if os.path.exists(REWARD_LOG_PATH):
        try:
            with open(REWARD_LOG_PATH, "r", encoding="utf-8") as file: return json.load(file)
        except Exception as e: print(f"[ERROR] Failed to load reward log: {e}")
    return []

def save_reward_log(reward_history):
    """Saves reward history to JSON file."""
    try:
        with open(REWARD_LOG_PATH, "w", encoding="utf-8") as file: json.dump(reward_history, file)
    except Exception as e: print(f"[ERROR] Failed to save reward log: {e}")

def plot_reward_history(reward_history):
    """Plots the total reward per episode."""
    if not reward_history:
        print("[WARN] No reward history data to plot.")
        return
    plt.figure(figsize=(10, 6)); plt.clf() # Clear previous figure data
    plt.plot(reward_history, label="Total Reward per Episode")
    if len(reward_history) >= 10:
        moving_avg = [sum(reward_history[max(0, i-10):i]) / min(i, 10) for i in range(1, len(reward_history) + 1)]
        plt.plot(moving_avg, label='10-Episode Moving Average', linestyle='--')
    plt.xlabel("Episodes"); plt.ylabel("Total Reward"); plt.title("AI Performance Over Time")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show(block=False); plt.pause(0.1)

def get_state_vector(game_state_dict):
    """Converts game state dictionary to a flat state vector."""
    if not game_state_dict: return None
    vision_grid = preprocess_vision_grid(game_state_dict.get("vision_grid", []))
    velocity = [game_state_dict.get("velocity_x", 0.0), game_state_dict.get("velocity_y", 0.0)]
    position = [game_state_dict.get("player_x", 0.0), game_state_dict.get("player_y", 0.0)]
    state_vector = vision_grid + velocity + position
    if len(state_vector) != INPUT_SIZE:
        print(f"[ERROR] State size mismatch! Expected {INPUT_SIZE}, got {len(state_vector)}. Check preprocessing.")
        state_vector = state_vector[:INPUT_SIZE]
        while len(state_vector) < INPUT_SIZE: state_vector.append(0.0)
    return state_vector

def decide_action(model, state_vector, epsilon):
    """Chooses action using epsilon-greedy strategy."""
    if random.random() < epsilon:
        action_index = random.choice(list(ACTION_MAP.keys()))
    else:
        if state_vector is None: return "idle"
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad(): action_values = model(state_tensor)
        model.train()
        action_index = torch.argmax(action_values, dim=1).item()
    return ACTION_MAP.get(action_index, "idle")

def train_model(model, optimizer, replay_memory, batch_size, gamma):
    """Trains the model using a batch from replay memory."""
    if len(replay_memory) < batch_size: return
    minibatch = random.sample(list(replay_memory), batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    state_tensors = torch.tensor(states, dtype=torch.float32)
    next_state_tensors = torch.tensor(next_states, dtype=torch.float32)
    reward_tensors = torch.tensor(rewards, dtype=torch.float32)
    action_indices = torch.tensor([REVERSE_ACTION_MAP[a] for a in actions], dtype=torch.int64)
    done_tensors = torch.tensor(dones, dtype=torch.bool)
    current_q_values = model(state_tensors).gather(1, action_indices.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = model(next_state_tensors)
        max_next_q = torch.max(next_q_values, dim=1)[0]
    target_q_values = reward_tensors + gamma * max_next_q * (~done_tensors)
    loss_fn = nn.MSELoss()
    loss = loss_fn(current_q_values, target_q_values)
    optimizer.zero_grad(); loss.backward(); optimizer.step()


# --- Main Execution ---
def run():
    print("[INFO] Initializing AI Runner...")
    # --- Initialization ---
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    load_model_state(model, MODEL_PATH)
    model.train()
    replay_memory = deque(maxlen=MEMORY_SIZE)
    reward_history = load_reward_log()
    total_episodes = len(reward_history)
    steps_done = 0

    # --- Calculate Epsilon based on loaded episodes ---
    epsilon = EPSILON_START
    for _ in range(total_episodes):
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    print(f"[INFO] Starting training. Resuming from Episode {total_episodes}.")
    print(f"[INFO] Resuming with calculated Epsilon: {epsilon:.4f}") # Log calculated epsilon

    print(f"[INFO] Ensure Godot is running and saving state to: {GAME_STATE_PATH}")
    print(f"[INFO] Using sync delay: {SYNC_DELAY_SECONDS}s between action and read.")
    print("-" * 30)
    current_episode_reward = 0.0
    steps_this_episode = 0
    last_saved_episode = total_episodes
    print("[INFO] Waiting for initial game state...")
    game_state = None
    while game_state is None:
        time.sleep(0.5); game_state = read_game_state()
        if not game_state: print("[WARN] Still waiting for initial game state...")
    s = get_state_vector(game_state)
    if s is None: print("[FATAL] Could not process initial game state. Exiting."); return
    print("[INFO] Initial state received. Starting main loop...")

    # --- Main Loop ---
    try:
        while True:
            action = decide_action(model, s, epsilon)
            save_action(action)
            time.sleep(SYNC_DELAY_SECONDS) # Wait for Godot
            next_game_state = read_game_state() # Uses retry logic now

            if next_game_state is None:
                print("[WARN] Failed to read next game state *after retries*. Skipping step, waiting longer...")
                time.sleep(0.5) # Longer wait if read failed completely
                continue

            reward = next_game_state.get("reward", 0.0)
            done = next_game_state.get("done", False)
            s_prime = get_state_vector(next_game_state)

            if s_prime is None:
                print("[WARN] Failed to process next game state vector. Skipping step.")
                game_state = next_game_state # Update dictionary anyway?
                continue

            replay_memory.append((s, action, reward, s_prime, done))
            s = s_prime; game_state = next_game_state
            current_episode_reward += reward; steps_done += 1; steps_this_episode += 1

            if steps_done % 200 == 0:
                 print(f"Step: {steps_done}, Ep: {total_episodes}, EpSteps: {steps_this_episode}, EpRew: {current_episode_reward:.2f}, Epsilon: {epsilon:.3f}, Mem: {len(replay_memory)}")

            train_model(model, optimizer, replay_memory, BATCH_SIZE, GAMMA)

            episode_timed_out = steps_this_episode >= MAX_STEPS_PER_EPISODE
            if done or episode_timed_out:
                end_reason = "Done Flag" if done else "Timeout"
                print("-" * 30); print(f"--- EPISODE {total_episodes} END ({end_reason} at step {steps_this_episode}) ---")
                print(f"   Total Reward: {current_episode_reward:.2f}")
                reward_history.append(current_episode_reward); print(f"   Reward history length: {len(reward_history)}")
                save_reward_log(reward_history) # Save log every episode

                # Decay Epsilon *after* using it for the episode's decisions
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY); print(f"   New Epsilon (for next ep): {epsilon:.4f}")

                if total_episodes > 0 and total_episodes % SAVE_EVERY_EPISODES == 0:
                     print(f"--- Saving Model & Plotting at Episode {total_episodes} ---")
                     save_model_state(model, MODEL_PATH)
                     print(f"   Plotting data (last 10): {reward_history[-10:]}")
                     plot_reward_history(reward_history); last_saved_episode = total_episodes

                # Prepare for next episode
                total_episodes += 1
                current_episode_reward = 0.0
                steps_this_episode = 0
                print("[INFO] Waiting for state after reset...")
                time.sleep(SYNC_DELAY_SECONDS * 2) # Give Godot time to reset and save
                game_state = None; read_attempts = 0
                while game_state is None and read_attempts < 10:
                    game_state = read_game_state() # Use retry read here too
                    if not game_state: print("[WARN] Still waiting for state after reset..."); time.sleep(0.2)
                    read_attempts += 1
                if game_state:
                    s = get_state_vector(game_state)
                    if s is None: print("[FATAL] Could not process state after reset. Exiting."); break
                    print(f"--- Starting Episode {total_episodes} ---")
                else: print("[FATAL] Failed to get state after reset. Exiting."); break
                print("-" * 30)

    except KeyboardInterrupt: print("\n[INFO] Training interrupted by user.")
    except Exception as e: print(f"\n!!!!!!!!!!!!!!!! PYTHON ERROR !!!!!!!!!!!!!!"); traceback.print_exc(); print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    finally:
        print("\n" + "-" * 30); print("[INFO] AI Runner stopping. Performing final save...")
        # Save model if new episodes completed since last save
        if total_episodes > 0 and total_episodes > last_saved_episode:
             save_model_state(model, MODEL_PATH)
             print(f"[INFO] Final model state saved to {MODEL_PATH}")
        save_reward_log(reward_history); print(f"[INFO] Final reward history saved (Length: {len(reward_history)}).")
        print("[INFO] Attempting final plot..."); plot_reward_history(reward_history)
        print("[INFO] Plot function called. Check plot window.")
        input("Press Enter to close the plot and exit...")

# --- Script Entry Point ---
if __name__ == "__main__":
    if not os.path.isdir(AI_DATA_FOLDER):
         print("*"*60); print(f"[FATAL ERROR] AI_DATA_FOLDER not found or not a directory:\n  '{AI_DATA_FOLDER}'\nPlease ensure this path is correct and points to Godot's user://AI_data location."); print("*"*60)
    else:
         print(f"[INFO] Using AI Data Folder: {AI_DATA_FOLDER}"); run()