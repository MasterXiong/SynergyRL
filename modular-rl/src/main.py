# from comet_ml import Experiment as Exp_co

import json
import os
import time

import numpy as np
import pymongo
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from tensorboardX import SummaryWriter
import gym

import checkpoint as cp
import utils
from algs import TD3
from arguments import get_args
from config import *
from vec_env.subproc_vec_env import SubprocVecEnv

# import wandb

ex = Experiment("synergy")

def setup_mongodb(db_url, db_name):
    # this function was taken from the pymarl distro
    client = None
    mongodb_fail = True

    for tries in range(5):
        # First try to connect to the central server. If that doesn't work then just save locally
        maxSevSelDelay = 10000  # Assume 10s maximum server selection delay
        try:
            print("Trying to connect to mongoDB '{}'".format(db_url))
            client = pymongo.MongoClient(
                db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay
            )
            client.server_info()
            ex.observers.append(
                MongoObserver.create(url=db_url, db_name=db_name, ssl=True)
            )  # db_name=db_name,
            print("Added MongoDB observer on {}.".format(db_url))
            mongodb_fail = False
            break
        except pymongo.errors.ServerSelectionTimeoutError:
            print("Couldn't connect to MongoDB on try {}".format(tries + 1))

    if mongodb_fail:
        print("Couldn't connect to MongoDB after 5 tries!")

    return client


@ex.main
def train(_run):
    # Set up directories ===========================================================
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
    exp_name = '/'.join([args.label, str(args.seed)])
    exp_path = os.path.join(DATA_DIR, exp_name)
    rb_path = os.path.join(BUFFER_DIR, exp_name)
    os.system(f'rm -r {exp_path}')
    os.system(f'rm -r {rb_path}')
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(rb_path, exist_ok=True)
    # save arguments
    with open(os.path.join(exp_path, "args.txt"), "w+") as f:
        json.dump(args.__dict__, f, indent=2)

    # Retrieve MuJoCo XML files for training ========================================
    envs_train_names = []
    args.unimal = 'unimal' in args.custom_xml
    args.graphs = dict()
    # existing envs
    if not args.custom_xml:
        for morphology in args.morphologies:
            envs_train_names += [
                name[:-4]
                for name in os.listdir(XML_DIR)
                if ".xml" in name and morphology in name
            ]
        for name in envs_train_names:
            args.graphs[name] = utils.getGraphStructure(
                os.path.join(XML_DIR, "{}.xml".format(name)),
                args.observation_graph_type,
            )
    # custom envs
    else:
        if os.path.isfile(args.custom_xml):
            assert ".xml" in os.path.basename(args.custom_xml), "No XML file found."
            name = os.path.basename(args.custom_xml)
            envs_train_names.append(name[:-4])  # truncate the .xml suffix
            args.graphs[name[:-4]] = utils.getGraphStructure(
                args.custom_xml, args.observation_graph_type
            )
        elif os.path.isdir(args.custom_xml):
            for name in os.listdir(args.custom_xml):
                if ".xml" in name:
                    envs_train_names.append(name[:-4])
                    args.graphs[name[:-4]] = utils.getGraphStructure(
                        os.path.join(args.custom_xml, name), args.observation_graph_type
                    )
        else:
            raise NotADirectoryError

    envs_train_names.sort()
    num_envs_train = len(envs_train_names)
    envs_test_names_set = [
        'walker_3_main','walker_6_main',
        'humanoid_2d_7_left_leg', 'humanoid_2d_7_right_arm',
        'cheetah_3_balanced', 'cheetah_5_back', 'cheetah_6_front',
    ]
    envs_test_names = [name for name in envs_train_names if name in envs_test_names_set]
    # ISSUE: the test envs are also included in training envs?
    args.envs_train_names = envs_train_names
    args.envs_test_names = envs_test_names
    print("#" * 50 + "\ntraining envs: {}\n".format(envs_train_names) + "#" * 50)
    print("#" * 50 + "\ntesting envs: {}\n".format(envs_test_names) + "#" * 50)

    # Set up training env and policy ================================================
    args.agent_obs_size, args.max_action = utils.registerEnvs(
        envs_train_names, args.max_episode_steps, args.custom_xml
    )
    print (f'limb obs size: {args.agent_obs_size}, max action: {args.max_action}')
    args.num_agents = {
        env_name: sum([len(x) for x in args.graphs[env_name]]) for env_name in envs_train_names
    }
    print ('limbs in each robot')
    print (args.num_agents)
    max_num_agents = max(args.num_agents.values())
    args.monolithic_max_agent = max_num_agents
    # create vectorized training env
    obs_max_len = (max_num_agents * args.agent_obs_size)
    # get limb names
    envs = [gym.make("environments:%s-v0" % env_name) for env_name in envs_train_names]
    args.limb_names = dict()
    for i, env in enumerate(envs):
        args.limb_names[envs_train_names[i]] = env.model.body_names[1:]
    # the wrapper add zero padding
    envs_train = [
        utils.makeEnvWrapper(name, obs_max_len, args.seed, unimal=args.unimal) for name in envs_train_names
    ]

    envs_train = SubprocVecEnv(envs_train)  # vectorized env
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # determine the maximum number of children in all the training envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(envs_train_names, args.graphs)

    args.max_num_agents = max_num_agents
    # args.start_timesteps = 0
    if args.enable_wandb:
        wandb.run.name = args.expID
        wandb.config.update(args)
    # setup agent policy
    policy = TD3.TD3(args)

    # Create new training instance or load previous checkpoint ========================
    has_ckp, model_name = cp.has_checkpoint(exp_path, rb_path)
    if has_ckp:
        print("*** loading checkpoint from {} ***".format(exp_path))
        (
            total_timesteps,
            episode_num,
            replay_buffer,
            num_samples,
            loaded_path,
        ) = cp.load_checkpoint(exp_path, rb_path, policy, args, model_name)
        print("*** checkpoint loaded from {} ***".format(loaded_path))
    else:
        print("*** training from scratch ***")
        # init training vars
        total_timesteps = 0
        episode_num = 0
        num_samples = 0
        # different replay buffer for each env; avoid using too much memory if there are too many envs
        replay_buffer = dict()
        if num_envs_train > args.rb_max // 1e6:
            for name in envs_train_names:
                replay_buffer[name] = utils.ReplayBuffer(
                    max_size=args.rb_max // num_envs_train
                )
        else:
            for name in envs_train_names:
                replay_buffer[name] = utils.ReplayBuffer()

    # Initialize training variables ================================================
    writer = SummaryWriter("%s/%s/" % (DATA_DIR, exp_name))
    s = time.time()
    timesteps_since_saving = 0
    timesteps_since_saving_model_only = 0
    this_training_timesteps = 0
    collect_done = True
    episode_timesteps_list = [0 for i in range(num_envs_train)]
    done_list = [True for i in range(num_envs_train)]

    # Start training ===========================================================

    # Trainging ================================================================
    model_savings_so_far = 0
    test_load_model_timesteps = total_timesteps
    logger_counter = 0
    while total_timesteps < args.max_timesteps:

        # train and log after one episode for each env
        if collect_done:
            # log updates and train policy
            if this_training_timesteps != 0:
                training_logs, extra_return = policy.train(
                    replay_buffer,
                    episode_timesteps_list,
                    args.batch_size,
                    args.discount,
                    args.tau,
                    args.policy_noise,
                    args.noise_clip,
                    args.policy_freq,
                    graphs=args.graphs,
                    envs_train_names=envs_train_names[:num_envs_train],
                    envs_test_names=envs_test_names,
                    extra_args={
                        'steps': total_timesteps
                    }
                )
                logger_counter  = (logger_counter + 1) % 10
                if logger_counter == 1:
                    # add to tensorboard display
                    policy_info = policy.get_info()
                    # logged_text=""
                    for k, v in policy_info.items():
                        if 'scaler' in k:
                            writer.add_scalar(k, v, total_timesteps)
                        else:
                            writer.add_embedding(mat=v, metadata=list(range(len(v))),
                                             global_step=total_timesteps, tag=f'synergy_weight_{k}')

                for k,v in training_logs.items():
                    if 'synergy_weight' in k:
                        v = v.cpu().numpy()
                        writer.add_embedding(mat=v, metadata=list(range(len(v))),
                                             global_step=total_timesteps, tag=k)
                    else:
                        writer.add_scalar(k,v, total_timesteps)
                        if args.enable_wandb:
                            wandb.log({k:v}, step=total_timesteps)
                for i in range(num_envs_train):
                    writer.add_scalar(
                        "{}_episode_reward".format(envs_train_names[i]),
                        episode_reward_list[i],
                        total_timesteps,
                    )
                    writer.add_scalar(
                        "{}_episode_len".format(envs_train_names[i]),
                        episode_timesteps_list[i],
                        total_timesteps,
                    )
                    writer.add_scalar(
                        "{}_action_mean".format(envs_train_names[i]),
                        np.mean(action_list[i]),
                        total_timesteps
                    )
                    writer.add_scalar(
                        "{}_action_std".format(envs_train_names[i]),
                        np.std(action_list[i]),
                        total_timesteps
                    )
                    if args.enable_wandb:
                        wandb.log({
                            "{}_episode_reward".format(envs_train_names[i]): episode_reward_list[i],
                            "{}_episode_len".format(envs_train_names[i]): episode_timesteps_list[i],
                            "{}_action_mean".format(envs_train_names[i]): np.mean(action_list[i]),
                            "{}_action_std".format(envs_train_names[i]): np.std(action_list[i]),
                        }, step=total_timesteps)
                    if not args.debug:
                        ex.log_scalar(
                            f"{envs_train_names[i]}_episode_reward",
                            float(episode_reward_list[i]),
                            total_timesteps,
                        )
                        ex.log_scalar(
                            f"{envs_train_names[i]}_episode_len",
                            float(episode_timesteps_list[i]),
                            total_timesteps,
                        )
                if not args.debug:
                    ex.log_scalar(
                        "total_timesteps", float(total_timesteps), total_timesteps,
                    )
                # print to console
                print(
                    "-" * 50
                    + "\nExpID: {}, FPS: {:.2f}, TotalT: {}, EpisodeNum: {}, SampleNum: {}, ReplayBSize: {}".format(
                        exp_name,
                        this_training_timesteps / (time.time() - s),
                        total_timesteps,
                        episode_num,
                        num_samples,
                        sum(
                            [
                                len(replay_buffer[name].storage)
                                for name in envs_train_names
                            ]
                        ),
                    )
                )
                for i in range(len(envs_train_names)):
                    print(
                        "{} === EpisodeT: {}, Reward: {:.2f}".format(
                            envs_train_names[i],
                            episode_timesteps_list[i],
                            episode_reward_list[i],
                        )
                    )

            # save model and replay buffers
            if timesteps_since_saving >= args.save_freq:
                timesteps_since_saving = 0
                model_saved_path = cp.save_model(
                    exp_path,
                    policy,
                    total_timesteps,
                    episode_num,
                    num_samples,
                    replay_buffer,
                    envs_train_names,
                    args,
                    model_name=f"model_{model_savings_so_far}.pyth",
                )
                model_savings_so_far += 1
                print("*** model saved to {} ***".format(model_saved_path))
                if args.save_buffer:
                    rb_saved_path = cp.save_replay_buffer(rb_path, replay_buffer)
                    print("*** replay buffers saved to {} ***".format(rb_saved_path))

            # reset training variables
            obs_list = envs_train.reset()
            done_list = [False for i in range(num_envs_train)]
            episode_reward_list = [0 for i in range(num_envs_train)]
            episode_timesteps_list = [0 for i in range(num_envs_train)]
            episode_num += num_envs_train
            # create reward buffer to store reward for one sub-env when it is not done
            episode_reward_list_buffer = [0 for i in range(num_envs_train)]

        # start sampling ===========================================================
        # sample action randomly for sometime and then according to the policy
        if total_timesteps < args.start_timesteps * num_envs_train:
            action_list = [
                np.random.uniform(
                    low=envs_train.action_space.low[0],
                    high=envs_train.action_space.high[0],
                    size=max_num_agents,
                )
                for i in range(num_envs_train)
            ]
        else:
            action_list = []
            for i in range(num_envs_train):
                # dynamically change the graph structure of the modular policy
                policy.change_morphology(args.graphs[envs_train_names[i]])
                # remove 0 padding of obs before feeding into the policy (trick for vectorized env)
                obs = np.array(
                    obs_list[i][
                        : args.agent_obs_size * args.num_agents[envs_train_names[i]]
                    ]
                )
                policy_action = policy.select_action(obs, envs_train_names[i])
                if args.expl_noise != 0:
                    policy_action = (
                        policy_action
                        + np.random.normal(0, args.expl_noise, size=policy_action.size)
                    ).clip(
                        envs_train.action_space.low[0], envs_train.action_space.high[0]
                    )
                # policy_action = policy_action.clip(
                #     envs_train.action_space.low[0], envs_train.action_space.high[0])
                # add 0-padding to ensure that size is the same for all envs
                policy_action = np.append(
                    policy_action,
                    np.array([0 for i in range(max_num_agents - policy_action.size)]),
                )
                action_list.append(policy_action)

        # perform action in the environment
        new_obs_list, reward_list, curr_done_list, _ = envs_train.step(action_list)

        # record if each env has ever been 'done'
        done_list = [done_list[i] or curr_done_list[i] for i in range(num_envs_train)]

        for i in range(num_envs_train):
            # add the instant reward to the cumulative buffer
            # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
            episode_reward_list_buffer[i] += reward_list[i]
            if curr_done_list[i] and episode_reward_list[i] == 0:
                episode_reward_list[i] = episode_reward_list_buffer[i]
                episode_reward_list_buffer[i] = 0
            done_bool = float(curr_done_list[i])
            if episode_timesteps_list[i] + 1 == args.max_episode_steps:
                done_bool = 0
                done_list[i] = True
            # remove 0 padding before storing in the replay buffer (trick for vectorized env)
            num_agents = args.num_agents[envs_train_names[i]]
            obs = np.array(obs_list[i][: args.agent_obs_size * num_agents])
            new_obs = np.array(new_obs_list[i][: args.agent_obs_size * num_agents])
            action = np.array(action_list[i][:num_agents])
            # insert transition in the replay buffer
            replay_buffer[envs_train_names[i]].add(
                (obs, new_obs, action, reward_list[i], done_bool)
            )
            num_samples += 1
            # do not increment episode_timesteps if the sub-env has been 'done'
            if not done_list[i]:
                episode_timesteps_list[i] += 1
                total_timesteps += 1
                this_training_timesteps += 1
                timesteps_since_saving += 1
                timesteps_since_saving_model_only += 1

        obs_list = new_obs_list
        collect_done = all(done_list)

    # save checkpoint after training ===========================================================
    model_saved_path = cp.save_model(
        exp_path,
        policy,
        total_timesteps,
        episode_num,
        num_samples,
        replay_buffer,
        envs_train_names,
        args,
    )
    print("*** training finished and model saved to {} ***".format(model_saved_path))
    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    args.enable_wandb = bool(args.enable_wandb)
    if args.enable_wandb:
        wandb.init(project="modular-rl")
    if not args.debug:
        # I was using mongodb to store the results, insert your credentials if you want to use it as well.
        # For code release, I switched that to a FileStorage observer, but I haven't tested it properly.
        # db_url = "mongodb://{user}:{pwd}@{DBSERVER}:{DBPORT}/{dbname}?authMechnism=SCRAM-SHA-1".format(
        #     user="sacred",
        #     pwd="sacredpassword",
        #     DBSERVER="localhost",
        #     DBPORT="27018",
        #     dbname="sacred",
        # )
        # # db_url = "mongodb://172.16.112.194:27018"
        # DBNAME = "sacred"
        # mongo_client = setup_mongodb(db_url, DBNAME)
        ex.observers.append(FileStorageObserver(f"{DATA_DIR}/{args.expID}"))
        ex.add_config(vars(args))
        ex.run()
    else:
        train()
