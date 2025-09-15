# End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing


## Introduction

End2Race is an end-to-end imitation learning framework for autonomous racing on the [F1Tenth platform](https://roboracer.ai/build). By learning from expert demonstrations generated through a Lattice Planner, the system captures temporal dependencies in racing dynamics to enable real-time control in competitive scenarios. End2Race addresses key challenges in autonomous racing—strategic planning, reactive control, and safe overtaking—through a unified neural network approach, demonstrating superior performance in both single-agent lap completion and multi-agent racing with high overtaking success rates and minimal collisions.

## Table of Contents
- [Code Structure](#code-structure)
- [Environment Setup](#environment-setup)
- [Evaluation](#evaluation)
- [Data Collection](#data-collection)
- [Training](#training)
- [Raceline Generation (Optional)](#raceline-generation-optional)

## Code Structure
```
end2race/
├── pretrained/
│   └── end2race.pth           # Pretrained model weights
├── f1tenth_gym/               # F1Tenth simulator environment
├── f1tenth_racetracks/        # Track data with pre-generated lanes and racelines
│   └── generate_raceline.py   # Raceline generation tool
├── latticeplanner/            # Expert planner module
│   ├── lattice_planner.py     # Main planner implementation
│   ├── lattice_config.yaml    # Planner configuration
│   ├── pure_pursuit.py        # Low-level trajectory tracker
│   └── utils.py               # Planner utility functions
├── model.py                   # GRU network architecture
├── train.py                   # Training script
├── expert.py                  # Lattice planner expert simulation
├── collect_dataset.py         # Batch data collection
├── evaluate_singleagent.py    # Single-agent lap completion evaluation
├── evaluate_multiagent.py     # Multi-agent competitive racing evaluation
├── evaluate_parallel.py       # Parallel batch evaluation
└── utils.py                   # Shared utility functions
```

## Environment Setup

### Base Requirements
* **Hardware**: 4-core CPU, 8GB RAM (GPU recommended for training and inference)
* **System**: Windows or Linux
* **Python**: 3.10 (Conda or native installation)

### Clone Repository
```bash
git clone https://github.com/li1164733168/end2race.git
cd end2race
```

### Setup Virtual Environment
```bash
conda create --name end2race python=3.10
conda activate end2race
```

### Install Dependencies
```bash
bash install.sh
```

## Evaluation

The evaluation is conducted using the [F1Tenth Gym simulator](https://github.com/f1tenth/f1tenth_gym), a high-fidelity racing environment for autonomous vehicle research. 

### Single-Agent Evaluation
Evaluates the model's lap completion ability across different track configurations, testing its robustness to varying track layouts and racing line complexities without opponent interaction.

```bash
python evaluate_singleagent.py \
    --model_path pretrained/end2race.pth \
    --map_name Austin \
    --noise 0.0 \
    --render
```
- `--model_path`: Path to trained model weights
- `--map_name`: Track name from f1tenth_racetracks
- `--noise`: Sensor noise level (0.0~1.0), fraction of LiDAR points masked
- `--render`: Enable visualization and video recording

### Multi-Agent Evaluation

Evaluates the model in competitive racing scenarios against an expert opponent. The framework provides 4 pre-configured tracks with raceline files for testing: [Austin](f1tenth_racetracks/Austin/Austin_map.png), [Hockenheim](f1tenth_racetracks/Hockenheim/Hockenheim_map.png), [MoscowRaceway](f1tenth_racetracks/MoscowRaceway/MoscowRaceway_map.png), and [Nuerburgring](f1tenth_racetracks/Nuerburgring/Nuerburgring_map.png), each with 3 raceline options (`raceline0`, `raceline1`, `raceline2`). For tracks without pre-generated racelines, use the [Raceline Generation](#raceline-generation-optional) section to create them first.


```bash
python evaluate_multiagent.py \
    --model_path pretrained/end2race.pth \
    --map_name Austin \
    --ego_idx 150 \
    --interval_idx 15 \
    --opp_raceline raceline0 \
    --opp_speedscale 0.5 \
    --sim_duration 8.0 \
    --noise 0.0 \
    --render
```
- `--ego_idx`: Starting waypoint index for ego vehicle
- `--interval_idx`: Waypoint offset between ego and opponent at start 
- `--opp_raceline`: Opponent's raceline choice
- `--opp_speedscale`: Opponent speed multiplier
- `--sim_duration`: Maximum simulation time in seconds 

### Multi-Agent Parallel Evaluation (Optional)

The batch evaluation runs hundreds of scenarios in parallel to comprehensively assess the model's performance across different starting positions, opponent strategies, and difficulty levels:

```bash
python evaluate_parallel.py \
    --model_path pretrained/end2race.pth \
    --map_name Austin \
    --num_workers 4 \
    --num_startpoints 50 \
    --oppo_racelines raceline0 raceline1 raceline2 \
    --oppo_speed_scales 0.5 0.6 0.7 0.8 \
    --sim_duration 8.0 \
    --noise 0.0 \
    --render
```
- `--num_workers`: Parallel processes for batch evaluation 
- `--num_startpoints`: Number of starting positions distributed along track
- `--oppo_racelines`: List of opponent racing lines
- `--oppo_speed_scales`: List of speed multipliers 


## Data Collection

### Single Collection

The single-agent scenario collects lap completion demonstrations where the Lattice Planner navigates the track without opponents:

```bash
python expert.py \
    --num_agents 1 \
    --map_name Austin \
    --ego_idx 0 \
    --sim_duration 8.0 \
    --render
```

The multi-agent scenario collects competitive racing demonstrations where the ego Lattice Planner must overtake or follow an opponent:

```bash
python expert.py \
    --num_agents 2 \
    --map_name Austin \
    --ego_idx 0 \
    --interval_idx 15 \
    --opp_raceline raceline0 \
    --opp_speed_scale 0.6 \
    --sim_duration 8.0 \
    --render
```
- `--num_agents`: 1 for single-agent racing, 2 for multi-agent racing
- `--ego_idx`: Starting waypoint index for ego vehicle 
- `--interval_idx`: Initial distance between vehicles in waypoints
- `--opp_raceline`: Opponent's raceline file 
- `--opp_speed_scale`: Opponent speed multiplier 
- `--sim_duration`: Simulation time limit in seconds

### Parallel Collection

The batch collection script automates the process by running multiple Lattice Planner simulations in parallel, systematically varying starting positions, opponent strategies, and speed settings to create a diverse training dataset:

```bash
python collect_dataset.py \
    --map_name Austin \
    --num_startpoints 50 \
    --opp_racelines raceline0 raceline1 raceline2 \
    --opp_speed_scales 0.5 0.6 0.7 0.8 \
    --sim_duration 8.0 \
    --workers 6 
```
- `--num_startpoints`: Number of starting positions distributed along track 
- `--opp_racelines`: List of opponent racing lines 
- `--opp_speed_scales`: List of speed multipliers 
- `--workers`: Number of parallel processes 


## Training
Trains the End2Race model using imitation learning on collected demonstrations.

```bash
python train.py \
    --data_path Dataset_Austin \
    --model_path end2race.pth \
    --hidden_scale 4 \
    --mask_prob 0.1 \
    --batch_size 16
```
- `--data_path`: Path to training data directory 
- `--model_path`: Path to save/load model weights
- `--hidden_scale`: GRU hidden size multiplier
- `--mask_prob`: Probability of masking speed input during training 
- `--batch_size`: Training batch size 

## Raceline Generation (Optional)

Generate optimized racing lines for new tracks. First, upload the track map files to `f1tenth_racetracks/{map_name}/` including `{map_name}_map.png` (binary image: white=drivable, black=walls) and `{map_name}_map.yaml` (map metadata). Then run:

```bash
cd f1tenth_racetracks
python generate_raceline.py \
    --map_name Austin \
    --num_lanes 3 \
    --v_max 7.5 \
    --inner_safe_dist 0.3 \
    --outer_safe_dist 0.3
```
- `--map_name`: Track name from f1tenth_racetracks
- `--num_lanes`: Number of lanes and racelines to generate 
- `--v_max`: Maximum velocity in m/s for optimization
- `--inner_safe_dist`: Safety margin from inner boundary 
- `--outer_safe_dist`: Safety margin from outer boundary 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.