from __future__ import print_function

import os
import subprocess
import pickle

from algs import TD3
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import checkpoint as cp
import utils
from arguments import get_args
from config import *
from gym.wrappers.monitoring import video_recorder
import gym

def evaluate(args, model_index=None, episode_num=10, model_interval=10):

    exp_name = '_'.join([args.label, str(args.seed)])
    exp_path = os.path.join(DATA_DIR, args.label, str(args.seed))

    if not os.path.exists(exp_path):
        raise FileNotFoundError("checkpoint does not exist")
    print("*** folder fetched: {} ***".format(exp_path))
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Retrieve MuJoCo XML files for evaluation ========================================
    env_names = []
    args.graphs = dict()
    args.unimal = 'unimal' in args.custom_xml
    # existing envs
    if not args.custom_xml:
        for morphology in args.morphologies:
            env_names += [
                name[:-4]
                for name in os.listdir(XML_DIR)
                if ".xml" in name and morphology in name
            ]
        for name in env_names:
            args.graphs[name] = utils.getGraphStructure(
                os.path.join(XML_DIR, "{}.xml".format(name)),
                args.observation_graph_type,
            )
    # custom envs
    else:
        if os.path.isfile(args.custom_xml):
            assert ".xml" in os.path.basename(args.custom_xml), "No XML file found."
            name = os.path.basename(args.custom_xml)
            env_names.append(name[:-4])  # truncate the .xml suffix
            args.graphs[name[:-4]] = utils.getGraphStructure(
                args.custom_xml, args.observation_graph_type
            )
        elif os.path.isdir(args.custom_xml):
            for name in os.listdir(args.custom_xml):
                if ".xml" in name:
                    env_names.append(name[:-4])
                    args.graphs[name[:-4]] = utils.getGraphStructure(
                        os.path.join(args.custom_xml, name), args.observation_graph_type
                    )

    env_names.sort()
    args.envs_train_names = env_names

    # Set up env and policy ================================================
    args.agent_obs_size, args.max_action = utils.registerEnvs(
        env_names, args.max_episode_steps, args.custom_xml
    )
    # determine the maximum number of children in all the envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(env_names, args.graphs)

    args.num_agents = {
        env_name: sum([len(x) for x in args.graphs[env_name]]) for env_name in env_names
    }
    args.max_num_agents = max(args.num_agents.values())
    args.monolithic_max_agent = args.max_num_agents

    envs = [gym.make("environments:%s-v0" % env_name) for env_name in env_names]
    args.limb_names = dict()
    for i, env in enumerate(envs):
        args.limb_names[env_names[i]] = env.model.body_names[1:]

    # setup agent policy
    policy = TD3.TD3(args)

    try:
        if model_index is None:
            model_files = [el for el in os.listdir(exp_path) if "model" in el]
            if 'model.pyth' in model_files:
                model_files.remove('model.pyth')
        elif model_index == 'final':
            model_files = ['model.pyth']
        else:
            model_files = [f'model_{model_index}.pyth']
    except:
        raise Exception(
            "policy loading failed; check policy params (hint 1: max_children must be the same as the trained policy; hint 2: did the trained policy use torchfold (consider pass --disable_fold)?"
        )
    model_files.sort()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # evaluate ===========================================================
    model_results = dict()
    for env_name in env_names:
        model_results[env_name] = dict()
        # create env
        env = utils.makeEnvWrapper(env_name, seed=args.seed, obs_max_len=None, unimal=args.unimal)()

        for idx in range(0, len(model_files)):
            print (idx)
            model_name = f'model_{idx}.pyth'
            cp.load_model_only(exp_path, policy, model_name)
            policy.change_morphology(args.graphs[env_name])

            obs = env.reset()
            done = False
            episode_reward = 0.
            avg_reward = 0.
            finished_episode_num = 0

            while (1):
                if done:
                    obs = env.reset()
                    done = False
                    # print(f'Episode Return: {episode_reward}')
                    avg_reward += episode_reward
                    episode_reward = 0.
                    finished_episode_num += 1
                    if finished_episode_num == episode_num:
                        break
                action = policy.select_action(np.array(obs), env_name)
                # perform action in the environment
                new_obs, reward, done, _ = env.step(action)
                episode_reward += reward
                obs = new_obs
        
            model_results[env_name][model_name.split('.')[0]] = avg_reward / episode_num
            print (f'{env_name}, {model_name}: {avg_reward / episode_num}')
        
        with open(f'{EVAL_DIR}/{exp_name}.pkl', 'wb') as f:
            pickle.dump(model_results, f)



if __name__ == "__main__":
    args = get_args()
    evaluate(args)
