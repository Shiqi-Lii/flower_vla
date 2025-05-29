import os
import torch
import numpy as np
from types import SimpleNamespace as Namespace
from flower_vla.eval.simpler.flower_inference_wrapper import UhaInference
from collections import defaultdict
import json
from datetime import datetime
from simpler_env.main_inference import main

import wandb

from flower_vla.eval.utils.utils import set_seed


def setup_environment(gpu_id="0"):
    """Setup CUDA environment and return device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Selected Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    return device

def setup_paths(debug_mode=True,):
    """Setup paths for data and logging."""
    maniskill_data_path = '/home/reuss/code/flower_vla_policy/SimplerEnv/ManiSkill2_real2sim/data'
    base_log_dir = "/home/reuss/code/flower_vla/results"
    log_dir = base_log_dir if not debug_mode else f"{base_log_dir}_debug"
    
    # Model paths
    return maniskill_data_path, log_dir

def get_task_configs(maniskill_data_path, log_dir, ckpt):
    """Define task configurations."""
    
    
    args_trace = {
        'medit_bridge': [
            Namespace(policy_setup='widowx_bridge', ckpt_path=ckpt, env_name='PutCarrotOnPlateInScene-v0', additional_env_save_tags=None, scene_name='bridge_table_1_v1', enable_raytracing=False, robot='widowx', obs_camera_name=None, action_scale=1.0, control_freq=5, sim_freq=500, max_episode_steps=60, rgb_overlay_path=f'{maniskill_data_path}/real_inpainting/bridge_real_eval_1.png', robot_init_x_range=[0.147, 0.147, 1.0], robot_init_y_range=[0.028, 0.028, 1.0], robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0], robot_init_rot_rpy_range=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], obj_variation_mode='episode', obj_episode_range=[0, 24], obj_init_x_range=[-0.35, -0.12, 5], obj_init_y_range=[-0.02, 0.42, 5], additional_env_build_kwargs=None, logging_dir=log_dir, tf_memory_limit=3072, octo_init_rng=0, robot_init_xs=np.array([0.147]), robot_init_ys=np.array([0.028]), robot_init_quats=[np.array([0., 0., 0., 1.], dtype=np.float32)]),
            Namespace(policy_setup='widowx_bridge', ckpt_path=ckpt, env_name='StackGreenCubeOnYellowCubeBakedTexInScene-v0', additional_env_save_tags=None, scene_name='bridge_table_1_v1', enable_raytracing=False, robot='widowx', obs_camera_name=None, action_scale=1.0, control_freq=5, sim_freq=500, max_episode_steps=60, rgb_overlay_path=f'{maniskill_data_path}/real_inpainting/bridge_real_eval_1.png', robot_init_x_range=[0.147, 0.147, 1.0], robot_init_y_range=[0.028, 0.028, 1.0], robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0], robot_init_rot_rpy_range=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], obj_variation_mode='episode', obj_episode_range=[0, 24], obj_init_x_range=[-0.35, -0.12, 5], obj_init_y_range=[-0.02, 0.42, 5], additional_env_build_kwargs=None, logging_dir=log_dir, tf_memory_limit=3072, octo_init_rng=0, robot_init_xs=np.array([0.147]), robot_init_ys=np.array([0.028]), robot_init_quats=[np.array([0., 0., 0., 1.], dtype=np.float32)]),
            Namespace(policy_setup='widowx_bridge', ckpt_path=ckpt, env_name='PutSpoonOnTableClothInScene-v0', additional_env_save_tags=None, scene_name='bridge_table_1_v1', enable_raytracing=False, robot='widowx', obs_camera_name=None, action_scale=1.0, control_freq=5, sim_freq=500, max_episode_steps=60, rgb_overlay_path=f'{maniskill_data_path}/real_inpainting/bridge_real_eval_1.png', robot_init_x_range=[0.147, 0.147, 1.0], robot_init_y_range=[0.028, 0.028, 1.0], robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0], robot_init_rot_rpy_range=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], obj_variation_mode='episode', obj_episode_range=[0, 24], obj_init_x_range=[-0.35, -0.12, 5], obj_init_y_range=[-0.02, 0.42, 5], additional_env_build_kwargs=None, logging_dir=log_dir, tf_memory_limit=3072, octo_init_rng=0, robot_init_xs=np.array([0.147]), robot_init_ys=np.array([0.028]), robot_init_quats=[np.array([0., 0., 0., 1.], dtype=np.float32)]),
            Namespace(policy_setup='widowx_bridge', ckpt_path=ckpt, env_name='PutEggplantInBasketScene-v0', additional_env_save_tags=None, scene_name='bridge_table_1_v2', enable_raytracing=False, robot='widowx_sink_camera_setup', obs_camera_name=None, action_scale=1.0, control_freq=5, sim_freq=500, max_episode_steps=120, rgb_overlay_path=f'{maniskill_data_path}/real_inpainting/bridge_sink.png', robot_init_x_range=[0.127, 0.127, 1.0], robot_init_y_range=[0.06, 0.06, 1.0], robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0], robot_init_rot_rpy_range=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], obj_variation_mode='episode', obj_episode_range=[0, 24], obj_init_x_range=[-0.35, -0.12, 5], obj_init_y_range=[-0.02, 0.42, 5], additional_env_build_kwargs=None, logging_dir=log_dir, tf_memory_limit=3072, octo_init_rng=0, robot_init_xs=np.array([0.127]), robot_init_ys=np.array([0.06]), robot_init_quats=[np.array([0., 0., 0., 1.], dtype=np.float32)]),
        ],
    }
    return args_trace

def process_args_for_debug(args_s, debug_settings):
    """Process arguments for debug mode."""
    num_confs, num_episodes, variation_limit = debug_settings
    args_s = args_s[:num_confs]
    
    variation_keys = [
        "robot_init_xs",
        "robot_init_ys",
        "robot_init_quats",
        "obj_init_xs",
        "obj_init_ys",
    ]
    
    for args in args_s:
        if hasattr(args, 'obj_episode_range'):
            minn = min(args.obj_episode_range[1], num_episodes)
            args.obj_episode_range = [0, minn]
            
        for key in variation_keys:
            if hasattr(args, key):
                setattr(args, key, getattr(args, key)[:variation_limit])
    
    return args_s

def run_evaluation(args_trace, model_base_dir, debug_mode=True, eval_config=None):
    """Run evaluation and collect results."""
    results = defaultdict(dict)
    all_successes = []
    
    for script, args_s in args_trace.items():
        print(f"\nEvaluating {script}...")
        
        if debug_mode:
            args_s = process_args_for_debug(args_s, (5, 1, 3))
            
        for args in args_s:
            model = UhaInference(
                saved_model_base_dir=model_base_dir,
                saved_model_path=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
                pred_action_horizon=eval_config["pred_action_horizon"] if eval_config else 5,
                multistep=eval_config["multistep"] if eval_config else 1,
                ensemble_strategy=eval_config["ensemble_strategy"] if eval_config else 'false',
                num_sampling_steps=eval_config["num_sampling_steps"] if eval_config else 5,
                use_ema=eval_config["use_ema"] if eval_config else True,
                use_torch_compile=eval_config["use_torch_compile"] if eval_config else False,
                use_dopri5=eval_config["use_dopri5"] if eval_config else False,
            )
            setattr(args, 'policy_model', 'flower_vla')
            
            print(f"Running {args.env_name}...")
            avg_success = main(args, model)  # Your main evaluation function
            
            results[script][args.env_name] = avg_success
            all_successes.append(avg_success)
            
            print(f"Task: {args.env_name}, Success Rate: {avg_success:.3f}")
            # Log to wandb as results come in
            wandb.log({
                f"{script}/{args.env_name}": avg_success,
                "current/running_mean": np.mean(all_successes),
                "current/tasks_completed": len(all_successes)
            })
    
    return results, all_successes

def save_results(results, all_successes, log_dir, ckpt):

    set_seed(42)
    """Save evaluation results with category averages and wandb logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(log_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate overall statistics
    avg_success = np.mean(all_successes)
    std_success = np.std(all_successes)
    
    # Calculate per-category averages
    category_averages = {}
    for category, tasks in results.items():
        category_success_rates = list(tasks.values())
        category_averages[category] = {
            'mean': float(np.mean(category_success_rates)),
            'std': float(np.std(category_success_rates)),
            'n_tasks': len(category_success_rates)
        }
    
    # Prepare results summary
    summary = {
        "timestamp": timestamp,
        "checkpoint": ckpt,
        "overall_average": float(avg_success),
        "overall_std": float(std_success),
        "category_averages": category_averages,
        "results_by_category": results
    }
    
    # Save results
    filename = os.path.join(results_dir, f"eval_results_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            "final/overall_success_rate": avg_success,
            "final/overall_std": std_success,
            **{f"final/{cat}_success_rate": stats['mean'] for cat, stats in category_averages.items()},
            **{f"final/{cat}_std": stats['std'] for cat, stats in category_averages.items()}
        })
        
        # Create and log a wandb Table with all results
        table_data = []
        for category, tasks in results.items():
            for task, success in tasks.items():
                table_data.append([category, task, success])
        
        results_table = wandb.Table(
            data=table_data,
            columns=["Category", "Task", "Success Rate"]
        )
        wandb.log({"final/results_table": results_table})
    
    # Print summary table
    print("\nEvaluation Summary:")
    print("=" * 50)
    print(f"Checkpoint: {ckpt}")
    print(f"Overall Average Success Rate: {avg_success:.3f} ± {std_success:.3f}")
    
    # Print category averages
    print("\nCategory Averages:")
    print("-" * 50)
    print("| Category | Tasks | Success Rate |")
    print("|----------|-------|--------------|")
    for category, stats in category_averages.items():
        print(f"| {category} | {stats['n_tasks']} | {stats['mean']:.3f} ± {stats['std']:.3f} |")
    
    print("\nResults by Task:")
    print("-" * 50)
    print("| Category | Task | Success Rate |")
    print("|----------|------|--------------|")
    for category, tasks in results.items():
        for task, success in tasks.items():
            print(f"| {category} | {task} | {success:.3f} |")
    
    print(f"\nDetailed results saved to: {filename}")
    return category_averages  # Return category averages for further use if needed

def main_evaluation():
    """Main evaluation function."""
    # Setup
    eval_config = {
        "pred_action_horizon": 10,
        "multistep": 5,
        "ensemble_strategy": 'false',
        "num_sampling_steps": 2,
        "use_ema": False,
        "use_torch_compile": True,
        "use_dopri5": False,
    }
    debug_mode = False  # Set to False for full evaluation

    # model_base_dir = "/home/reuss/code/flower_vla_policy/horeka_trains/"
    model_base_dir = '/home/reuss/code/flower_vla/logs/runs/2025-02-06/'
    # model_base_dir = '/home/reuss/code/flower_vla/horeka_trains/'
    # ckpt = '07-02-36/checkpoint_360000'  # Checkpoint to evaluate
    ckpt = '12-12-25/checkpoint_160000'
    ckpt = '08-25-43/checkpoint_480000'
    ckpt = '23-01-18/checkpoint_360000'
    ckpt = '16-21-07/checkpoint_280000'
    # ckpt = '13-10-38/checkpoint_220000'
    # ckpt = '05-04-39/checkpoint_180000'
    # Initialize wandb
    run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="flower_bridge_simpler_evaluation",
        name=run_name,
        config={
            "eval_config": eval_config,
            "debug_mode": debug_mode,
            "checkpoint": ckpt
        }
    )
    
    print("Starting evaluation with config:")
    print(f"Debug mode: {debug_mode}")
    print("Model settings:", json.dumps(eval_config, indent=2))

    device = setup_environment()
    maniskill_data_path, log_dir = setup_paths(debug_mode)
    
    # Get task configurations
    args_trace = get_task_configs(maniskill_data_path, log_dir, ckpt)
    
    # Run evaluation
    results, all_successes = run_evaluation(args_trace, model_base_dir, debug_mode, eval_config=eval_config)
    
    # Save and display results
    save_results(results, all_successes, log_dir, ckpt)

if __name__ == "__main__":
    main_evaluation()