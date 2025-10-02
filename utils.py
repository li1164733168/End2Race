import os
import numpy as np

def load_raceline_with_speed(map_name, raceline_file, start_idx):
    """Load raceline waypoints with position and speed information"""
    raceline_path = f"f1tenth_racetracks/{map_name}/{raceline_file}"
    with open(raceline_path, 'r') as f:
        lines = f.readlines()[1:]
    
    waypoints = []
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 6:
            waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    waypoints = np.array(waypoints)
    
    # Get starting position and speed
    idx = start_idx % len(waypoints)
    start_pose = np.array([[waypoints[idx, 0], waypoints[idx, 1], waypoints[idx, 2]]])
    initial_speed = waypoints[idx, 3]
    
    return start_pose, initial_speed, waypoints

def calculate_metrics(trajectory, speeds):
    """Calculate performance metrics"""
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    total_distance = sum(np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i]))
                        for i in range(len(trajectory)-1)) if len(trajectory) > 1 else 0
    return avg_speed, speed_variance, total_distance

def create_render_callback(render_info, visited_points, drawn_points, batch_objects, lap_num):
    """Create render callback function for visualization"""
    from pyglet.gl import GL_POINTS
    
    def render_callback(e):
        x, y = e.cars[0].vertices[::2], e.cars[0].vertices[1::2]
        e.left, e.right = min(x) - 800, max(x) + 800
        e.top, e.bottom = max(y) + 800, min(y) - 800
        e.score_label.x, e.score_label.y = e.left + 800, e.top - 1500
        
        e.score_label.text = (f"Laps: {render_info['laps']}/{lap_num} | "
                            f"Time: {render_info['lap_time']:.1f}s | "
                            f"Speed: {render_info['speed']:.1f}m/s | "
                            f"Steer: {render_info['steer']:.2f}rad")
        
        # Draw trajectory
        for i, pt in enumerate(visited_points):
            x, y = 50.0 * pt[0], 50.0 * pt[1]
            if i < len(drawn_points):
                drawn_points[i].vertices = [x, y, 0.0]
            else:
                b = e.batch.add(1, GL_POINTS, None,
                              ('v3f/stream', [x, y, 0.0]),
                              ('c3B/stream', [0, 0, 255]))
                drawn_points.append(b)
                batch_objects.append(b)
    
    return render_callback

def find_corresponding_waypoint(ego_waypoint, opp_waypoints):
    """Find the waypoint on opponent raceline closest to ego waypoint spatially"""
    ego_position = ego_waypoint[:2]
    distances = np.linalg.norm(opp_waypoints[:, :2] - ego_position, axis=1)
    return np.argmin(distances)

def load_positions_and_speeds_from_params(params, map_name):
    """Load initial positions and speeds based on segment parameters (from run_lattice_planner.py)"""
    base_path = f"f1tenth_racetracks/{map_name}"
    
    # Load ego raceline with speed
    ego_path = os.path.join(base_path, params['ego_raceline'] + '.csv')
    with open(ego_path, 'r') as f:
        lines = f.readlines()[1:]
    ego_waypoints = []
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 6:
            ego_waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    ego_waypoints = np.array(ego_waypoints)
    
    # Load opponent raceline with speed
    opp_path = os.path.join(base_path, params['opp_raceline'] + '.csv')
    if params['opp_raceline'] != params['ego_raceline']:
        with open(opp_path, 'r') as f:
            lines = f.readlines()[1:]
        opp_waypoints = []
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 6:
                opp_waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
        opp_waypoints = np.array(opp_waypoints)
    else:
        opp_waypoints = ego_waypoints
    
    # Get positions and speeds using direct indices
    ego_idx = params['ego_idx'] % len(ego_waypoints)
    opp_idx = params['opp_idx'] % len(opp_waypoints)
    positions = np.array([ego_waypoints[ego_idx, :3], opp_waypoints[opp_idx, :3]])
    initial_speeds = np.array([ego_waypoints[ego_idx, 3], opp_waypoints[opp_idx, 3]])
    
    return positions, initial_speeds

def get_ego_idx_range(map_name, ego_raceline, num_startpoints):
    """Generate evenly distributed evaluation points"""
    raceline_path = os.path.join('f1tenth_racetracks', map_name, ego_raceline)
    waypoints = np.loadtxt(raceline_path, delimiter=';', skiprows=1)
    max_waypoints = len(waypoints)
    ego_idx_range = np.linspace(0, max_waypoints - 1, num_startpoints, dtype=int).tolist()
    return ego_idx_range