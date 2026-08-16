import json
import neat
import time
import os
import pickle

GAME_STATE_FILE = "C://Users//LOQ//Documents//AI RUNNER//AI_data//game_state.json"
ACTION_FILE = "C://Users//LOQ//Documents//AI RUNNER//AI_data//ai_action.json"
BEST_GENOME_FILE = "C://Users//LOQ//Documents//AI RUNNER//AI_data//best_genome.pkl"

def read_game_state():
    """ Reads the game state and vision grid from the JSON file. """
    if not os.path.exists(GAME_STATE_FILE):
        print("⚠ No game state file found.")
        return None

    try:
        with open(GAME_STATE_FILE, "r") as file:
            state = json.load(file)
            vision_grid = state.get("vision_grid", [])

            # Flatten vision grid if needed
            if isinstance(vision_grid, list) and isinstance(vision_grid[0], list):
                vision_grid = [cell for row in vision_grid for cell in row]

            # Ensure vision grid has 49 values
            if len(vision_grid) != 49:
                print(f"⚠ Warning: Vision grid size mismatch! Expected 49, got {len(vision_grid)}")
                vision_grid = [0] * 49  # Default to blank grid if incorrect size

            # Ensure other required keys exist
            state.setdefault("player_x", 0)
            state.setdefault("player_y", 0)
            state.setdefault("velocity_x", 0)
            state.setdefault("velocity_y", 0)

            state["vision_grid"] = vision_grid
            return state

    except (json.JSONDecodeError, FileNotFoundError):
        print("❌ Error reading game state.")
        return None

def save_action(action):
    """ Saves the chosen AI action to the JSON file and ensures it's written. """
    try:
        with open(ACTION_FILE, "w") as file:
            json.dump({"action": action}, file)
            file.flush()  # Ensure data is written
            os.fsync(file.fileno())  # Force write to disk
    except IOError as e:
        print(f"❌ Error saving AI action: {e}")

def load_best_genome(config):
    """ Loads the best trained genome from file. """
    if os.path.exists(BEST_GENOME_FILE):
        with open(BEST_GENOME_FILE, "rb") as f:
            genome = pickle.load(f)
        return neat.nn.FeedForwardNetwork.create(genome, config)
    return None

def decide_action_with_neat(net, state):
    """ Uses the trained NEAT neural network to decide the next action. """
    if not state or not isinstance(state, dict):
        state = {"player_x": 0, "player_y": 0, "velocity_x": 0, "velocity_y": 0, "vision_grid": [0]*49}

    # Ensure vision_grid exists
    vision_grid = state.get("vision_grid", [0] * 49)  # Default to empty grid if missing
    inputs = vision_grid + [state["player_x"], state["player_y"], state["velocity_x"], state["velocity_y"]]

    print(f"🔍 Expected Inputs: {len(net.input_nodes)} | Provided Inputs: {len(inputs)}")  # LOG INPUT LENGTH
    print(f"📊 Input Values: {inputs}")  # LOG ACTUAL INPUT DATA

    if len(inputs) != 53:
        raise ValueError(f"❌ Input size mismatch! Expected 53, got {len(inputs)}.")

    output = net.activate(inputs)

    actions = ["left", "right", "jump", "idle"]
    chosen_action = actions[output.index(max(output))]

    return chosen_action


def run_trained_ai():
    """ Loads the trained AI and plays using it. """
    config_path = r"C:\Users\LOQ\Documents\AI RUNNER\AI_data\neat_config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    net = load_best_genome(config)

    if not net:
        print("❌ No trained AI found! Run `neat_runner.py` first.")
        return

    print("✅ Loaded trained AI! Playing now...")

    while True:
        state = read_game_state()
        if state:
            action = decide_action_with_neat(net, state)
            save_action(action)

        time.sleep(0.2)  # Slow down to match game loop

if __name__ == "__main__":
    run_trained_ai()
