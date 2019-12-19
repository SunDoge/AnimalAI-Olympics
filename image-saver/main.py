from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import random
import sys
import argparse
from typed_args import TypedArgs
import numpy as np
from PIL import Image
from pathlib import Path
import os
import imageio
from tqdm import tqdm

env_path = '../env/AnimalAI'
worker_id = random.randint(0, 200)
run_seed = 1
no_graphics = True


class Args(TypedArgs):

    def __init__(self):
        parser = argparse.ArgumentParser()
        self.arena_config = parser.add_argument(
            '-a', '--arena_config', default='../examples/configs/allObjectsRandom.yaml'
        )
        self.output = parser.add_argument(
            '-o', '--output'
        )

        self.parse_args_from(parser)


def init_environment(env_path, no_graphics, worker_id, seed, docker_target_name=None):
    if env_path is not None:
        # Strip out executable extensions if passed
        env_path = (env_path.strip()
                    .replace('.app', '')
                    .replace('.exe', '')
                    .replace('.x86_64', '')
                    .replace('.x86', ''))
    # docker_training = docker_target_name is not None
    docker_training=False

    return UnityEnvironment(
        n_arenas=1,
        file_name=env_path,
        worker_id=worker_id,
        seed=seed,
        docker_training=docker_training,
        # inference=False,
        play=False
    )


def main(args: Args):
    arena_config_in = ArenaConfig(args.arena_config)
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    env = init_environment(env_path, no_graphics, worker_id, run_seed)
    env.reset(arenas_configurations=arena_config_in)

    num_arenas = 10
    num_images_per_arena = 10

    for arena_id in tqdm(range(num_arenas)):
        env.reset(arenas_configurations=arena_config_in)
        for image_id in tqdm(range(num_images_per_arena)):
            # res = env.step(vector_action=np.random.randint(0, 3, size=2 * 1))
            res = env.step(vector_action=[0, 1])

            # visual_observation: List[ndarray[NUM_ARENAS, 84, 84, 3]]
            np_image = res['Learner'].visual_observations[0][0]
            # import ipdb; ipdb.set_trace()


            np_image = (np_image * 255).astype(np.uint8)
            imageio.imsave(
                output_dir / f'{arena_id}-{image_id}.jpg', np_image
            )


if __name__ == "__main__":
    main(Args())
