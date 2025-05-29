"""DROID language annotation transform"""
import json
import tensorflow as tf
from typing import Dict, Optional, Any

class DROIDLanguageTransform:
    def __init__(self, annotations_path: str):
        """Initialize with annotations path"""
        self.annotations = {}
        with tf.io.gfile.GFile(annotations_path, 'r') as f:
            self.annotations = json.load(f)
    
    def __call__(self, traj: dict) -> dict:
        """Make the transform callable as required by dlimp"""
        # Get episode path from trajectory metadata 
        filepath = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        
        # Extract episode ID from filepath
        try:
            episode_id = filepath.split('/')[-1].split('_')[-1].split('.')[0]
            annotations = self.annotations.get(episode_id)
            
            if annotations:
                # Add all three language annotations
                traj["task"]["language_instruction"] = tf.constant(annotations[0]) 
                traj["task"]["language_instruction_2"] = tf.constant(annotations[1])
                traj["task"]["language_instruction_3"] = tf.constant(annotations[2])
                
                # Update pad mask dict to show we have valid language instructions
                traj["task"]["pad_mask_dict"]["language_instruction"] = tf.ones_like(
                    traj["task"]["pad_mask_dict"]["language_instruction"], 
                    dtype=tf.bool
                )
            
        except Exception as e:
            print(f"Error processing episode {filepath}: {e}")
            
        return traj