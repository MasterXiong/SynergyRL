from __future__ import print_function

import os
import subprocess

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

def generate_video(args):

    total_time = args.video_length * 100
    # exp_name = args.expID
    exp_name = '_'.join([args.label, str(args.seed)])
    exp_path = os.path.join(DATA_DIR, args.label, str(args.seed))

    if not os.path.exists(exp_path):
        raise FileNotFoundError("checkpoint does not exist")
    print("*** folder fetched: {} ***".format(exp_path))
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Retrieve MuJoCo XML files for visualizing ========================================
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
        model_files = [el for el in os.listdir(exp_path) if "model" in el]
        if len(model_files) == 1:
            model_name = model_files[0]
        else:
            model_files.remove('model.pyth')
            max_version = max(
                [int(el.split("_")[1].split(".")[0]) for el in model_files]
            )
            model_name = f"model_{max_version}.pyth"
        print (model_name)
        cp.load_model_only(exp_path, policy, model_name)
    except:
        raise Exception(
            "policy loading failed; check policy params (hint 1: max_children must be the same as the trained policy; hint 2: did the trained policy use torchfold (consider pass --disable_fold)?"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # visualize ===========================================================
    for env_name in env_names:
        # create env
        env = utils.makeEnvWrapper(env_name, seed=args.seed, obs_max_len=None, unimal=args.unimal)()
        # create unique temp frame dir
        count = 0
        frame_dir = os.path.join(
            VIDEO_DIR, "frames_{}_{}_{}".format(exp_name, env_name, count)
        )
        # if args.unimal:
        #     from environments.unimal.config import cfg
        #     from environments.unimal.envs.vec_env.vec_video_recorder import VecVideoRecorder
        #     env = VecVideoRecorder(
        #         env,
        #         frame_dir,
        #         record_video_trigger=lambda x: x == 0,
        #         video_length=cfg.PPO.VIDEO_LENGTH,
        #         # file_prefix=self.file_prefix,
        #     )
        policy.change_morphology(args.graphs[env_name])

        while os.path.exists(frame_dir):
            count += 1
            frame_dir = "{}/frames_{}_{}_{}".format(
                VIDEO_DIR, exp_name, env_name, count
            )
        os.makedirs(frame_dir)
        # create video name without overwriting previously generated videos
        count = 0
        video_name = "%s_%s_%d" % ("".join(exp_name), env_name, count)
        while os.path.exists("{}/{}.mp4".format(VIDEO_DIR, video_name)):
            count += 1
            video_name = "%s_%s_%d" % (
                "".join(exp_name),
                env_name,
                count,
            )

        # init env vars
        done = True
        print("-" * 50)
        time_step_counter = 0
        printProgressBar(0, total_time)
        episode_reward = None

        while time_step_counter < total_time:
            printProgressBar(time_step_counter + 1, total_time, prefix=env_name)
            if done:
                obs = env.reset()
                done = False
                if episode_reward:
                    print("Episode Return: " + str(episode_reward))
                episode_reward = 0
            action = policy.select_action(np.array(obs), env_name)
            # perform action in the environment
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            # draw image of current frame
            if args.unimal:
                image_data = env.render(mode='rgb_array')
                # video_recorder.VideoRecorder(
                #     env=env, base_path=frame_dir, metadata={"step_id": time_step_counter}
                # ).capture_frame()
            else:
                image_data = env.sim.render(
                    VIDEO_RESOLUATION[0], VIDEO_RESOLUATION[1], camera_name="track"
                )# ('patrol_view', 'tune_camera', 'side', 'incline_view', 'mani_view', 'obstacle_view', 'ft_view', 'vt_view', 'left_view', 'top_down', 'front_view', 'rear_view').
            img = Image.fromarray(image_data, "RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("./misc/sans-serif.ttf", 24)
            font_small = ImageFont.truetype("./misc/sans-serif.ttf", 16)
            draw.text(
                (200, 10), "Instant Reward: " + str(reward), (255, 0, 0), font=font
            )
            draw.text(
                (200, 35),
                "Episode Reward: " + str(episode_reward),
                (255, 0, 0),
                font=font,
            )
            for i in range(len(action)):
                draw.text(
                    (10, 60 + 20*i),
                    f"({i},{obs.reshape(-1, 19)[i, -3]:>8.4f},{action[i]:>8.4f},{new_obs.reshape(-1, 19)[i, -3]:>8.4f})",
                    (0, 255, 0),
                    font=font_small
                )
            img.save(os.path.join(frame_dir, "frame-%.10d.png" % time_step_counter))

            obs = new_obs
            time_step_counter += 1

        # redirect output so output does not show on window
        FNULL = open(os.devnull, "w")
        # create video
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "50",
                "-y",
                "-i",
                os.path.join(frame_dir, "frame-%010d.png"),
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                os.path.join(VIDEO_DIR, "{}.mp4".format(video_name)),
            ],
            stdout=FNULL,
            stderr=subprocess.STDOUT,
        )
        subprocess.call(["rm", "-rf", frame_dir])


# Print iterations progress
# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(
    iteration,
    total,
    prefix="Video Progress:",
    suffix="Complete",
    decimals=1,
    length=35,
    fill="█",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)

    print("\r%s | %s%% %s" % (prefix, percent, suffix), end="\r")

    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    args = get_args()
    generate_video(args)
