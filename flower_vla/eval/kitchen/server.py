"""
A server for hosting a FlowerVLA model for inference.

On action server: pip install uvicorn fastapi json-numpy
On client: pip install requests json-numpy

On client:

import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()

Reset and provide the task before starting the rollout:

requests.post("http://serverip:port/reset", json={"text": ...})

Sample an action:

action = loads(
    requests.post(
        "http://serverip:port/query",
        json={"observation": ...},
    ).json()
)
"""


import logging
import os
import random
import hydra
import json_numpy
from omegaconf import DictConfig, OmegaConf
import torch

from flower.agents.utils.diffuser_ema import EMAModel
from flower.eval.kitchen.inference_wrapper import UhaInference as KitchenUhaInference

json_numpy.patch()
from collections import deque
import time
import traceback
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn
from accelerate import Accelerator


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))

class FlowerServer:
    def __init__(self, cfg):
        whole_path = os.path.join(cfg.train_run_dir, cfg.checkpoint)
        splits = whole_path.split('/')
        base_dir = "/".join(splits[:-2]) + '/'
        path = "/".join(splits[-2:])

        single_image = cfg.get('single_image', False)

        self.model = KitchenUhaInference(
            saved_model_base_dir=base_dir,
            saved_model_path=path,
            policy_setup="skip_policy_setup",
            pred_action_horizon=cfg.pred_action_horizon,
            multistep=cfg.multistep,
            ensemble_strategy=cfg.ensemble_strategy if cfg.ensemble_strategy else 'false',
            num_sampling_steps=cfg.num_sampling_steps,
            use_ema=cfg.use_ema,
            use_torch_compile=cfg.use_torch_compile,
            use_dopri5=cfg.use_dopri5,
            single_image=single_image
        )

        self.text = None

    def run(self, port=8000, host="0.0.0.0"):
        self.app = FastAPI()
        self.app.post("/query")(self.sample_actions)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset(self, payload: Dict[Any, Any]):
        self.text = payload['text']

        return "reset"

    def sample_actions(self, payload: Dict[Any, Any]):
        # payload needs to contain primary_image, secondary_image

        assert self.text is not None

        if "ensemble" in payload and payload["ensemble"]:
            self.model.ensemble_strategy = payload['ensemble']
        if "multistep" in payload and payload["multistep"]:
            self.model.ensemble_strategy = payload['multistep']

        try:
            action = self.model.step(primary_image=payload['primary_image'], secondary_image=payload['secondary_image'], task_description=self.text)

            return json_response(action)
        except:
            print(traceback.format_exc())
            return "error"

@hydra.main(config_path="../../../conf/eval", config_name="kitchen_server")
def main(cfg):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    server = FlowerServer(cfg)
    server.run(host="0.0.0.0", port=8003)


if __name__ == "__main__":
    main()
