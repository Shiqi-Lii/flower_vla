import threading
from collections import OrderedDict
import pickle
import torch

class AdvancedLangEmbeddingBuffer:
    def __init__(self, language_encoder, robot_info_buffer_size=200, goal_instruction_buffer_size=10000):
        self.language_encoder = language_encoder
        self.robot_info_buffer_size = robot_info_buffer_size
        self.goal_instruction_buffer_size = goal_instruction_buffer_size
        self.robot_info_buffer = OrderedDict()
        self.goal_instruction_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()

    def get_or_encode_batch(self, buffer, tokenized_texts, buffer_size):
        try:
            with self.buffer_lock:
                # Convert tensor to tuple for hashing
                tokenized_texts_tuples = [tuple(t.flatten().tolist()) for t in tokenized_texts]
                uncached_indices = [i for i, t in enumerate(tokenized_texts_tuples) if t not in buffer]
            
            if uncached_indices:
                uncached_tokenized = tokenized_texts[uncached_indices]
                encoded_batch = self.language_encoder(uncached_tokenized)
                
                for i, embedding in zip(uncached_indices, encoded_batch):
                    self.add_to_buffer(buffer, tokenized_texts_tuples[i], embedding, buffer_size)
            
            with self.buffer_lock:
                encoded_texts = [buffer[tuple(t.flatten().tolist())] for t in tokenized_texts]
            
            return torch.stack(encoded_texts)

        except Exception as e:
            print(f"Error encoding texts: {e}")
            return torch.zeros((len(tokenized_texts), 1, 512))

    def add_to_buffer(self, buffer, key, value, max_size):
        with self.buffer_lock:
            if len(buffer) >= max_size:
                buffer.popitem(last=False)
            buffer[key] = value

    def get_robot_info_embedding(self, robot_info):
        return self.get_or_encode_batch(self.robot_info_buffer, robot_info.unsqueeze(0), self.robot_info_buffer_size)

    def get_goal_instruction_embedding(self, goal_instruction):
        return self.get_or_encode_batch(self.goal_instruction_buffer, goal_instruction.unsqueeze(0), self.goal_instruction_buffer_size)

    def get_robot_info_embeddings(self, robot_infos):
        return self.get_or_encode_batch(self.robot_info_buffer, robot_infos, self.robot_info_buffer_size)

    def get_goal_instruction_embeddings(self, goal_instructions):
        return self.get_or_encode_batch(self.goal_instruction_buffer, goal_instructions, self.goal_instruction_buffer_size)

    def clear_buffers(self):
        with self.buffer_lock:
            self.robot_info_buffer.clear()
            self.goal_instruction_buffer.clear()

    def get_buffer_sizes(self):
        with self.buffer_lock:
            return {
                "robot_info_buffer": len(self.robot_info_buffer),
                "goal_instruction_buffer": len(self.goal_instruction_buffer)
            }

    def preload_common_tensors(self, robot_info_list, goal_instruction_list):
        self.get_or_encode_batch(self.robot_info_buffer, robot_info_list, self.robot_info_buffer_size)
        self.get_or_encode_batch(self.goal_instruction_buffer, goal_instruction_list, self.goal_instruction_buffer_size)

    def save_buffers(self, filepath):
        with self.buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'robot_info': self.robot_info_buffer,
                    'goal_instruction': self.goal_instruction_buffer
                }, f)

    def load_buffers(self, filepath):
        with open(filepath, 'rb') as f:
            buffers = pickle.load(f)
        with self.buffer_lock:
            self.robot_info_buffer = OrderedDict(list(buffers['robot_info'].items())[-self.robot_info_buffer_size:])
            self.goal_instruction_buffer = OrderedDict(list(buffers['goal_instruction'].items())[-self.goal_instruction_buffer_size:])