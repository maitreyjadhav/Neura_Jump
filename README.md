Neura_Jump

Neura_Jump is a Godot-based AI platformer where an AI agent learns to navigate the environment and reach objectives through machine learning.

The project combines Godot for the game environment and simulation with Python for training and controlling the AI agent.

Features
Godot-based game environment
AI-controlled player
Neural network-based decision making
AI training through evolutionary learning
Reward-based learning
Trained model saving and loading
Python ↔ Godot communication
Real-time AI gameplay
Tech Stack
Godot — Game environment and simulation
Python — AI training and inference
PyTorch — Neural network/model
JSON — Communication and AI data
Git/GitHub — Version control
How It Works

The project has two main components:

Godot

Godot handles:

Game physics
Player movement
Environment
Obstacles
Rewards
Game state
Communication with the Python AI
Python

Python handles:

AI training
Neural network
Decision making
Model saving/loading
Training data

The general pipeline is:
Godot Environment
       ↓
   Game State
       ↓
    Python AI
       ↓
 Neural Network
       ↓
    Action
       ↓
   Godot Player
       ↓
      Reward
       ↓
   AI Training

Project Structure
   Neura_Jump/
│
├── assets/                  # Game assets
├── scenes/                  # Godot scenes
├── scripts/                 # Godot scripts
├── AI_data/                 # AI-related data
│
├── project.godot            # Godot project configuration
├── model.py                 # Neural network/model
├── model.pth                # Trained model
├── neat_runner.py           # AI training
├── neat_runner_backup.py    # Backup training script
├── play_trained_ai.py       # Run trained AI
├── ai_action.json           # AI ↔ Godot communication
├── reward_log.json           # Reward/training logs
└── requirements.txt         # Python dependencies
