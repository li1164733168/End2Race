import os
import json
import argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from typing import Dict
from tqdm import tqdm
import subprocess
from utils import *

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch evaluation of End2Race models')
    parser.add_argument('--model_path', type=str, default='pretrained/end2race.pth')
    parser.add_argument('--hidden_scale', type=int, default=4)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--map_name', type=str, default='Hockenheim')
    
    # New arguments
    parser.add_argument('--render', action='store_true', help='Enable rendering for all evaluations')
    parser.add_argument('--sim_duration', type=float, default=8.0)
    parser.add_argument('--ego_raceline', type=str, default='raceline1')
    parser.add_argument('--opp_racelines', type=str, nargs='+', default=['raceline0', 'raceline1', 'raceline2'])
    parser.add_argument('--opp_speed_scales', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--interval_idx', type=int, default=15)
    parser.add_argument('--num_startpoints', type=int, default=5)
    
    return parser.parse_args()

def run_eval_subprocess(cmd):
    """Run evaluate_multiagent.py as subprocess and parse results"""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    metrics = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            if key in ['AVG_SPEED', 'TOTAL_DISTANCE', 'SPEED_VARIANCE']:
                metrics[key] = float(value)
            elif key == 'STATE':
                metrics[key] = int(value)
            else:
                metrics[key] = value
    
    # Determine category based on exit code
    if result.returncode == 1:
        category = 'following'
    elif result.returncode == 2:
        category = 'overtaking'
    elif result.returncode == 3:
        category = 'collision'
    elif result.returncode == 4:
        category = 'uturn'
    else:
        category = 'error'
    
    metrics['CATEGORY'] = category
    metrics['EXIT_CODE'] = result.returncode
    
    if result.returncode not in [1, 2, 3, 4] and result.stderr:
        metrics['ERROR'] = result.stderr.strip()[:200]
    
    return metrics

def generate_evaluation_report(results, model_path, map_name, noise_level, params):
    """Generate comprehensive evaluation report"""
    total = len(results)
    
    # Categorize results
    categories = {'following': 0, 'overtaking': 0, 'collision': 0, 'uturn': 0, 'error': 0}
    for r in results:
        cat = r.get('CATEGORY', 'error')
        if cat in categories:
            categories[cat] += 1
    
    # Calculate metrics for successful runs
    successful_results = [r for r in results if r.get('CATEGORY') in ['following', 'overtaking']]
    
    performance_metrics = {}
    if successful_results:
        avg_speeds = [r.get('AVG_SPEED', 0) for r in successful_results if r.get('AVG_SPEED') is not None]
        speed_variances = [r.get('SPEED_VARIANCE', 0) for r in successful_results if r.get('SPEED_VARIANCE') is not None]
        distances = [r.get('TOTAL_DISTANCE', 0) for r in successful_results if r.get('TOTAL_DISTANCE') is not None]
        
        performance_metrics = {
            'avg_speed': np.mean(avg_speeds) if avg_speeds else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'std_speed': np.std(avg_speeds) if avg_speeds else 0,
            'std_distance': np.std(distances) if distances else 0,
            'avg_speed_variance': np.mean(speed_variances) if speed_variances else 0,
            'std_speed_variance': np.std(speed_variances) if speed_variances else 0
        }
    
    report = {
        'model_path': model_path,
        'model_type': 'speed',  # Always speed model now
        'map_name': map_name,
        'noise_level': noise_level,
        'total_segments': total,
        
        'summary': {
            'following': categories['following'],
            'overtaking': categories['overtaking'],
            'collision': categories['collision'],
            'uturn': categories['uturn'],
            'error': categories['error'],
            'success_rate': (categories['following'] + categories['overtaking']) / total * 100 if total > 0 else 0,
            'collision_rate': categories['collision'] / total * 100 if total > 0 else 0,
            'uturn_rate': categories['uturn'] / total * 100 if total > 0 else 0
        },
        
        'performance_metrics': performance_metrics,
        'evaluation_params': params
    }
    
    return report

def evaluate_segment_wrapper(params: Dict) -> Dict:
    """Wrapper for multiprocessing pool"""
    
    # Build command for evaluate_multiagent.py
    cmd = [
        'python', 'evaluate_multiagent.py',
        '--model_path', params['model_path'],
        '--map_name', params['map_name'],
        '--ego_idx', str(params['ego_idx']),
        '--interval_idx', str(params['interval_idx']),
        '--ego_raceline', params['ego_raceline'],
        '--opp_raceline', params['opp_raceline'],
        '--opp_speedscale', str(params['opp_speed_scale']),
        '--sim_duration', str(params['sim_duration']),
        '--hidden_scale', str(params['hidden_scale']),
        '--noise', str(params['noise_level'])
    ]
    
    # Add render flag if enabled
    if params['render']:
        cmd.append('--render')
    
    metrics = run_eval_subprocess(cmd)
    metrics['PARAMS'] = params
    
    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    
    # Use argument values instead of hardcoded values
    ego_raceline = args.ego_raceline
    opp_racelines = args.opp_racelines
    opp_speed_scales = args.opp_speed_scales
    interval_idx = args.interval_idx
    num_startpoints = args.num_startpoints
    
    # Generate evaluation points
    ego_idx_range = get_ego_idx_range(args.map_name, f"{ego_raceline}.csv", num_startpoints)
    
    # Generate all parameter combinations
    all_params = []
    for ego_idx in ego_idx_range:
        for opp_raceline in opp_racelines:
            for speed_scale in opp_speed_scales:
                all_params.append({
                    'model_path': args.model_path,
                    'map_name': args.map_name,
                    'hidden_scale': args.hidden_scale,
                    'noise_level': args.noise,
                    'ego_idx': ego_idx,
                    'interval_idx': interval_idx,
                    'ego_raceline': ego_raceline,
                    'opp_raceline': opp_raceline,
                    'opp_speed_scale': speed_scale,
                    'sim_duration': args.sim_duration,
                    'render': args.render
                })
    
    total_segments = len(all_params)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting batch evaluation of {total_segments} segments")
    print(f"Model: {args.model_path}")
    print(f"Map: {args.map_name}")
    print(f"Workers: {args.num_workers}")
    print(f"Noise level: {args.noise}")
    
    start_time = datetime.now()
    
    # Run evaluations with progress bar
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(evaluate_segment_wrapper, all_params),
            total=total_segments,
            desc="Evaluating segments"
        ))
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nEvaluation complete in {elapsed:.1f} seconds")
    
    # Print category summary
    categories = {}
    for r in results:
        cat = r.get('CATEGORY', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nResults by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} ({count/total_segments*100:.1f}%)")
    
    # Generate report
    evaluation_params = {
        'ego_raceline': ego_raceline,
        'opp_racelines': opp_racelines,
        'opp_speed_scales': opp_speed_scales,
        'interval_idx': interval_idx,
        'num_startpoints': num_startpoints
    }

    report = generate_evaluation_report(results, args.model_path, args.map_name, args.noise, evaluation_params)

    # Save results
    output_data = {'report': report, 'detailed_results': results}

    # Create the same directory structure as videos: eval_results/{model_name}_{map_name}{noise_str}/
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    noise_str = f"_noise{int(args.noise*100)}" if args.noise > 0 else ""
    output_dir = os.path.join("eval_results", f"{model_name}_{args.map_name}{noise_str}")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"Collision rate: {report['summary']['collision_rate']:.1f}%")
    print(f"U-turn rate: {report['summary']['uturn_rate']:.1f}%")