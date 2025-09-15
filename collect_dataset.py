import os
import argparse
import glob
import subprocess
import numpy as np
import json
import re
from tqdm import tqdm
from multiprocessing import Pool
from itertools import product
from datetime import datetime
from typing import List, Dict

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch collect and validate lattice planner data')

    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--map_name', type=str, default='Austin')
    parser.add_argument('--ego_racelines', type=str, nargs='+', default=['raceline1'])
    parser.add_argument('--opp_racelines', type=str, nargs='+', default=['raceline0', 'raceline1', 'raceline2'])
    parser.add_argument('--opp_speed_scales', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--interval_idxs', type=int, nargs='+', default=[15])
    parser.add_argument('--sim_duration', type=float, default=8.0)
    parser.add_argument('--num_startpoints', type=int, default=50)
    
    return parser.parse_args()

class LatticeDataCollector:
    """Handles parallel batch data collection for lattice planner"""
    
    def __init__(self, num_workers: int = 6, save_video: bool = False, map_name: str = 'Austin',
                 ego_racelines: List[str] = None, opp_racelines: List[str] = None, opp_speed_scales: List[float] = None,
                 interval_idxs: List[int] = None, sim_duration: float = None, num_startpoints: int = None):
        self.num_workers = num_workers
        self.save_video = save_video
        self.map_name = map_name
        self.ego_racelines = ego_racelines 
        self.opp_racelines = opp_racelines 
        self.opp_speed_scales = opp_speed_scales 
        self.interval_idxs = interval_idxs 
        self.sim_duration = sim_duration
        self.num_startpoints = num_startpoints
        self.ego_idx_range = self._get_ego_idx_range()
    
    def _get_ego_idx_range(self) -> List[int]:
        """Generate evenly distributed starting points"""
        raceline_path = os.path.join('f1tenth_racetracks', self.map_name, f"{self.ego_racelines[0]}.csv")
        waypoints = np.loadtxt(raceline_path, delimiter=';', skiprows=2)
        max_waypoints = len(waypoints)
        ego_idx_range = np.linspace(0, max_waypoints - 1, self.num_startpoints, dtype=int).tolist()
        return ego_idx_range
        
    def generate_parameter_combinations(self) -> List[Dict]:
        """Generate all parameter combinations for simulations"""
        combinations = []
        for ego_raceline, opp_raceline, opp_speed, interval_idx, ego_idx in product(
            self.ego_racelines, self.opp_racelines, self.opp_speed_scales, 
            self.interval_idxs, self.ego_idx_range
        ):
            combinations.append({
                'ego_raceline': ego_raceline,
                'opp_raceline': opp_raceline,
                'opp_speed_scale': opp_speed,
                'interval_idx': interval_idx,
                'ego_idx': ego_idx
            })
        return combinations
    
    def run_single_simulation(self, params: Dict):
        """Run a single lattice planner simulation with given parameters"""
        cmd = [
            'python', 'expert.py',
            '--num_agents', '2',
            '--map_name', self.map_name,
            '--raceline', params['ego_raceline'],        
            '--opp_raceline', params['opp_raceline'],
            '--opp_speed_scale', str(params['opp_speed_scale']),
            '--ego_idx', str(params['ego_idx']),
            '--interval_idx', str(params['interval_idx']),
            '--sim_duration', str(self.sim_duration)
        ]
        
        if self.save_video:
            cmd.append('--render')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'data saved to' in line.lower() or 'video saved to' in line.lower():
                    match = re.search(rf'(Dataset_{self.map_name}_\d{{4}})', line)
                    if match:
                        return True, "Success", match.group(1)
            
            dirs = glob.glob(f"Dataset_{self.map_name}_*")
            if dirs:
                latest_dir = max(dirs, key=os.path.getctime)
                return True, "Success", latest_dir
            return False, "No output directory found", None
        else:
            return False, f"Failed: {result.stderr}", None
    
    def collect_data_parallel(self) -> List[str]:
        """Run lattice planner simulations in parallel with progress bar"""
        param_combinations = self.generate_parameter_combinations()
        total_jobs = len(param_combinations)
        
        print(f"Lattice Planner Data Collection")
        print(f"Map: {self.map_name}")
        print(f"Total jobs: {total_jobs}")
        print(f"Workers: {self.num_workers}")
        
        output_dirs = set()
        success_count = 0
        failed_count = 0
        
        with Pool(processes=self.num_workers) as pool:
            for i, (success, message, output_dir) in enumerate(
                tqdm(pool.imap(self.run_single_simulation, param_combinations),
                    total=total_jobs, desc="Running simulations")
            ):
                if success:
                    success_count += 1
                    if output_dir:
                        output_dirs.add(output_dir)
                else:
                    failed_count += 1
        
        print(f"\nSuccessful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Output directories created: {len(output_dirs)}")
        
        return list(output_dirs)

class DataValidator:
    """Handles data validation and report generation"""
    
    def __init__(self, input_dir: str, collector_params: Dict = None):
        self.input_dir = input_dir
        self.collector_params = collector_params
        self.report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_simulations": 0,
                "success_count": 0,
                "collision_count": 0,
                "error_count": 0
            },
            "success_details": {
                "follow_count": 0,
                "overtake_count": 0,
                "follow_cases": [],
                "overtake_cases": []
            },
            "collision_details": {
                "collision_cases": []
            },
            "error_details": {
                "error_reasons": {},
                "error_cases": []
            }
        }
    
    def parse_filename(self, filename: str) -> Dict:
        """Parse lattice planner filename: f_ol1_e150_o165_s0.8.csv"""
        basename = filename.replace('.csv', '').replace('.mp4', '').replace('.json', '')
        parts = basename.split('_')
        
        state = 'follow' if parts[0] == 'f' else 'overtake'
        opp_raceline = parts[1].replace('ol', 'raceline') + '.csv'
        ego_idx = int(parts[2].replace('e', ''))
        opp_idx = int(parts[3].replace('o', ''))
        speed_scale = float(parts[4].replace('s', ''))
        
        return {
            'state': state,
            'ego_raceline': 'raceline1.csv',
            'ego_idx': ego_idx,
            'opp_raceline': opp_raceline,
            'opp_idx': opp_idx,
            'speed_scale': speed_scale
        }
    
    def parse_collision_metadata(self) -> List[Dict]:
        """Parse collision metadata JSON files"""
        collision_dir = os.path.join(self.input_dir, "collision")
        collision_cases = []
        
        if os.path.exists(collision_dir):
            metadata_files = glob.glob(os.path.join(collision_dir, "*.json"))
            for metadata_path in metadata_files:
                with open(metadata_path, 'r') as f:
                    collision_case = json.load(f)
                    collision_cases.append(collision_case)
        
        return collision_cases
    
    def check_csv_file(self, file_path: str):
        """Check a single CSV file for validity"""
        import pandas as pd
        df = pd.read_csv(file_path)
        
        if len(df.columns) != 363:
            return False, f"Wrong column count: {len(df.columns)}", None
        
        expected_cols = ['time', 'steer', 'desired_speed'] + [f'lidar_{i}' for i in range(360)]
        if list(df.columns) != expected_cols:
            return False, "Column names mismatch", None
        if df.isna().any().any():
            return False, f"{df.isna().sum().sum()} NaN values", None
        if df['steer'].min() < -0.52 or df['steer'].max() > 0.52:
            return False, "Steering out of range", None
        if df['desired_speed'].min() < 0.1 or df['desired_speed'].max() > 8.5:
            return False, "Speed out of range", None
        
        lidar_cols = [f"lidar_{i}" for i in range(360)]
        if (df[lidar_cols] < 0).any().any():
            return False, "Negative lidar values", None
        if (df[lidar_cols] > 100).any().any():
            return False, "Lidar values > 100", None
        
        return True, "OK", df
    
    def validate_and_organize_files(self):
        """Validate all files, move invalid ones to error folder, and generate report"""
        success_dir = os.path.join(self.input_dir, "success")
        collision_dir = os.path.join(self.input_dir, "collision")
        error_dir = os.path.join(self.input_dir, "error")
        os.makedirs(error_dir, exist_ok=True)
        
        success_csv_files = []
        if os.path.exists(success_dir):
            success_csv_files = glob.glob(os.path.join(success_dir, "[fo]_ol*_e*_o*_s*.csv"))
        
        print(f"\nValidating files in {self.input_dir}...")
        print(f"  Success CSV files: {len(success_csv_files)}")
        
        valid_count = 0
        moved_count = 0
        
        for csv_file in success_csv_files:
            filename = os.path.basename(csv_file)
            is_valid, error_msg, df = self.check_csv_file(csv_file)
            
            if is_valid:
                valid_count += 1
                file_info = self.parse_filename(filename)
                file_info['interval_idx'] = self.collector_params['interval_idx']
                
                if file_info['state'] == 'follow':
                    self.report_data['success_details']['follow_count'] += 1
                    self.report_data['success_details']['follow_cases'].append(file_info)
                else:
                    self.report_data['success_details']['overtake_count'] += 1
                    self.report_data['success_details']['overtake_cases'].append(file_info)
            else:
                import shutil
                error_csv_path = os.path.join(error_dir, filename)
                shutil.move(csv_file, error_csv_path)
                moved_count += 1
                
                error_type = error_msg.split(':')[0]
                if error_type not in self.report_data['error_details']['error_reasons']:
                    self.report_data['error_details']['error_reasons'][error_type] = 0
                self.report_data['error_details']['error_reasons'][error_type] += 1
                
                file_info = self.parse_filename(filename)
                file_info['error_reason'] = error_msg
                file_info['interval_idx'] = self.collector_params['interval_idx']
                self.report_data['error_details']['error_cases'].append(file_info)
        
        collision_cases = self.parse_collision_metadata()
        self.report_data['collision_details']['collision_cases'] = collision_cases
        
        self.report_data['summary']['success_count'] = valid_count
        self.report_data['summary']['collision_count'] = len(collision_cases)
        self.report_data['summary']['error_count'] = moved_count
        self.report_data['summary']['total_simulations'] = valid_count + len(collision_cases) + moved_count
        
        print(f"  Valid CSV files: {valid_count}")
        print(f"  Invalid files moved to error/: {moved_count}")
        print(f"  Collision cases: {len(collision_cases)}")
        
        report_path = os.path.join(self.input_dir, "report.json")
        with open(report_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        print(f"  Report saved to: {report_path}")
        print(f"\nReport Summary:")
        print(f"  Total simulations: {self.report_data['summary']['total_simulations']}")
        print(f"  Successful: {self.report_data['summary']['success_count']} (Follow: {self.report_data['success_details']['follow_count']}, Overtake: {self.report_data['success_details']['overtake_count']})")
        print(f"  Collisions: {self.report_data['summary']['collision_count']}")
        print(f"  Errors: {self.report_data['summary']['error_count']}")

if __name__ == "__main__":
    """Main execution function"""
    args = parse_arguments()
    
    total_jobs = (len(args.ego_racelines) * len(args.opp_racelines) * len(args.opp_speed_scales) * len(args.interval_idxs) * args.num_startpoints)
    
    print(f"\nLattice Planner Batch Data Collection")
    print(f"=====================================")
    print(f"Map: {args.map_name}")
    print(f"Ego racelines: {args.ego_racelines}")
    print(f"Opponent racelines: {args.opp_racelines}")
    print(f"Speed scales: {args.opp_speed_scales}")
    print(f"Intervals: {args.interval_idxs}")
    print(f"Time per run: {args.sim_duration}s")
    print(f"Starting points: {args.num_startpoints}")
    print(f"Total jobs: {total_jobs}")
    print(f"Workers: {args.workers}")
    print(f"Video recording: {'Enabled' if args.render else 'Disabled'}")

    collector = LatticeDataCollector(
        num_workers=args.workers,
        save_video=args.render,
        map_name=args.map_name,
        ego_racelines=args.ego_racelines,
        opp_racelines=args.opp_racelines,
        opp_speed_scales=args.opp_speed_scales,
        interval_idxs=args.interval_idxs,
        sim_duration=args.sim_duration,
        num_startpoints=args.num_startpoints
    )
    output_dirs = collector.collect_data_parallel()
    
    for output_dir in output_dirs:
        print(f"\nValidating and generating report for: {output_dir}")
        validator = DataValidator(output_dir, collector_params={'interval_idx': args.interval_idxs[0] if args.interval_idxs else 15})
        validator.validate_and_organize_files()