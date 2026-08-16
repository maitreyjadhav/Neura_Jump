import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Constants
INPUT_SIZE = 356  # Vision Grid (352) + Velocity (2) + Position (2)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4  # Actions: Left, Right, Jump, Idle

MODEL_PATH = "model.pth"
AI_DATA_FOLDER = "C://Users//LOQ//Documents//AI RUNNER//AI_data"
GAME_STATE_PATH = os.path.join(AI_DATA_FOLDER, "game_state.json")
AI_ACTION_PATH = os.path.join(AI_DATA_FOLDER, "ai_action.json")
REWARD_LOG_PATH = "reward_log.json"

ACTION_MAP = {0: "left", 1: "right", 2: "jump", 3: "idle"}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

# AI Memory for Tracking Coins
MEMORY = set()

# Preprocess vision grid
def preprocess_vision_grid(vision_grid):
    processed_grid = []
    expected_rows, expected_cols = 22, 16

    for row_idx, row in enumerate(vision_grid[:expected_rows]):  # First 22 rows
        for col_idx, val in enumerate(row[:expected_cols]):  # First 16 columns
            processed_grid.append({
                0: 0.0,  # Empty space
                1: -1.0,  # Wall/Obstacle
                2: 1.0,  # Coin
                3: 0.5   # AI's position
            }.get(val, 0.0))  # Default 0.0 for unknown

            # Store coin positions in memory
            if val == 2:
                MEMORY.add((row_idx, col_idx))

    while len(processed_grid) < 352:
        processed_grid.append(0.0)  # Padding
    return processed_grid[:352]  # Trim if too large

# Load and save model
def load_model():
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    return model

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)

# Read game state
def read_game_state():
    if not os.path.exists(GAME_STATE_PATH):
        print("[WARNING] Game state file not found.")
        return {}
    try:
        with open(GAME_STATE_PATH, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content:
                print("[ERROR] game_state.json is empty!")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON Decode Error: {e}")
        return {}

# Save AI action
def save_action(action):
    with open(AI_ACTION_PATH, "w") as file:
        json.dump({"action": action}, file)

# Load and save reward history
def load_reward_log():
    if os.path.exists(REWARD_LOG_PATH):
        with open(REWARD_LOG_PATH, "r") as file:
            return json.load(file)
    return []

def save_reward_log(reward_history):
    with open(REWARD_LOG_PATH, "w") as file:
        json.dump(reward_history, file)

def plot_reward_history(reward_history):
    if len(reward_history) > 0:
        plt.plot(reward_history, label="Total Reward per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("AI Performance Over Time")
        plt.legend()
        plt.show()

# Determine best action
def decide_action(model, game_state, epsilon=0.1):
    vision_grid = preprocess_vision_grid(game_state.get("vision_grid", [[0] * 16 for _ in range(22)]))
    velocity = [game_state.get("velocity_x", 0), game_state.get("velocity_y", 0)]
    position = [game_state.get("player_x", 0), game_state.get("player_y", 0)]
    
    state = vision_grid + velocity + position

    if len(state) != 356:
        print(f"[ERROR] State size mismatch! Expected 356, but got {len(state)}")
        return "idle"
    
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Exploration (random moves)
    if random.random() < epsilon:
        return random.choice(["left", "right", "jump", "idle"])

    # Neural network decision
    with torch.no_grad():
        action_values = model(state_tensor)

    action_index = torch.argmax(action_values).item()
    chosen_action = ACTION_MAP.get(action_index, "idle")

    # Use memory to move toward unseen coins
    if 1.0 not in vision_grid:  # No coins in sight
        if MEMORY:
            target_x, target_y = next(iter(MEMORY))  # Pick a remembered coin
            if target_x > position[0]:
                return "right"
            elif target_x < position[0]:
                return "left"
            elif target_y < position[1]:
                return "jump"

    return chosen_action

# Train AI model
def train(model, optimizer, memory, gamma=0.9):
    if len(memory) < 10:
        return

    batch = random.sample(memory, 10)
    loss_fn = nn.MSELoss()

    for state, action, reward, next_state in batch:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        target = model(state_tensor).detach()

        # Encourage collecting coins
        if 1.0 in state and 1.0 not in next_state:  
            reward += 20  
        
        # Encourage moving toward coins
        if 1.0 in state:  
            reward += 8  

        # Penalize hitting walls
        if -1.0 in next_state:
            reward -= 5  

        # Discourage idling when a coin is visible
        if action == "idle" and 1.0 in state:
            reward -= 3  

        # Predict future reward
        with torch.no_grad():
            next_q_values = model(next_state_tensor)
            max_next_q = torch.max(next_q_values).item()

        action_index = REVERSE_ACTION_MAP[action]
        target[action_index] = reward + gamma * max_next_q

        optimizer.zero_grad()
        output = model(state_tensor)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

# Run AI loop
def run():
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = []
    reward_history = load_reward_log()
    episode = len(reward_history)  

    print("[INFO] AI Runner Started...")

    while True:
        game_state = read_game_state()
        if not game_state:
            print("[WARNING] No game state found.")
            continue

        state = preprocess_vision_grid(game_state.get("vision_grid", [[0] * 16 for _ in range(22)])) + \
                [game_state.get("velocity_x", 0), game_state.get("velocity_y", 0)] + \
                [game_state.get("player_x", 0), game_state.get("player_y", 0)]

        action = decide_action(model, game_state)
        save_action(action)
        print(f"[INFO] Chosen Action: {action}")

        reward = game_state.get("reward", 0)
        memory.append((state, action, reward, state))  
        train(model, optimizer, memory)
        save_model(model)

        if game_state.get("done", False):
            reward_history.append(reward)
            save_reward_log(reward_history)
            if episode % 10 == 0:
                plot_reward_history(reward_history)
            episode += 1

if __name__ == "__main__":
    run()
