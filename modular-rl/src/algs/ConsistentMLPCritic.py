import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConsistentMLPCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_children,
            disable_fold,
            td,
            bu,
            args=None,
    ):
        super(ConsistentMLPCritic, self).__init__()
        self.num_agents = 1
        self.x1 = [None] * self.num_agents
        self.x2 = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.input_action = [None] * self.num_agents
        self.msg_down = [None] * self.num_agents
        self.msg_up = [None] * self.num_agents
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.monolithic_max_agent = args.monolithic_max_agent
        self.limb_names = args.limb_names
        self.limb_num = dict()
        for env_name in self.limb_names:
            self.limb_num[env_name] = len(self.limb_names[env_name])
        if 'humanoid' in list(self.limb_names.keys())[0]:
            full_limbs = self.limb_names['humanoid_2d_9_full']
        elif 'hopper' in list(self.limb_names.keys())[0]:
            full_limbs = self.limb_names['hopper_5']
        elif 'walker' in list(self.limb_names.keys())[0]:
            full_limbs = self.limb_names['walker_7_main']
        self.limb_idx = dict()
        for env_name in self.limb_names:
            self.limb_idx[env_name] = []
            for limb in self.limb_names[env_name]:
                idx = full_limbs.index(limb)
                self.limb_idx[env_name].append(idx)

        self.critic1 = nn.Sequential(
            nn.Linear( (self.state_dim + action_dim) * self.monolithic_max_agent, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim * self.monolithic_max_agent)
        ).to(device)

        self.critic2 = nn.Sequential(
            nn.Linear((self.state_dim + action_dim) * self.monolithic_max_agent, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim * self.monolithic_max_agent)
        ).to(device)

    def forward(self, state, action, synergy, env_name):
        self.clear_buffer()
        batch_size = state.shape[0]

        self.input_state = state.reshape(batch_size, self.num_agents, -1)
        self.input_action = action.reshape(batch_size, self.num_agents, -1)

        # inpt = torch.cat([self.input_state, self.input_action], dim=2).reshape(batch_size, -1)
        # inpt = F.pad(inpt,
        #              pad=[0, (self.state_dim + self.action_dim) * (self.monolithic_max_agent - self.num_agents)],
        #              value = 0
        #              )
        compact_inpt = torch.cat([self.input_state, self.input_action], dim=2)
        inpt = torch.zeros(batch_size, self.monolithic_max_agent, compact_inpt.size(-1), device=device)
        inpt[:, self.limb_idx[env_name], :] = compact_inpt
        inpt = inpt.reshape(batch_size, -1)
        # self.x1 = self.critic1(inpt)[:, :self.num_agents].squeeze(-1)
        # self.x2 = self.critic2(inpt)[:, :self.num_agents].squeeze(-1)
        self.x1 = self.critic1(inpt)[:, self.limb_idx[env_name]].squeeze(-1)
        self.x2 = self.critic2(inpt)[:, self.limb_idx[env_name]].squeeze(-1)
        return self.x1, self.x2

    def Q1(self, state, action, synergy, env_name):
        self.clear_buffer()
        batch_size = state.shape[0]
        self.input_state = state.reshape(batch_size, self.num_agents, -1)
        self.input_action = action.reshape(batch_size, self.num_agents, -1)
        # inpt = torch.cat([self.input_state, self.input_action], dim=2).reshape(batch_size, -1)
        # inpt = F.pad(inpt,
        #              pad=[0, (self.state_dim + self.action_dim) * (self.monolithic_max_agent - self.num_agents)],
        #              value=0
        #              )
        compact_inpt = torch.cat([self.input_state, self.input_action], dim=2)
        inpt = torch.zeros(batch_size, self.monolithic_max_agent, compact_inpt.size(-1), device=device)
        inpt[:, self.limb_idx[env_name], :] = compact_inpt
        inpt = inpt.reshape(batch_size, -1)
        # self.x1 = self.critic1(inpt)[:, :self.num_agents].squeeze(-1)
        self.x1 = self.critic1(inpt)[:, self.limb_idx[env_name]].squeeze(-1)
        return self.x1


    def clear_buffer(self):
        self.x1 = [None] * self.num_agents
        self.x2 = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.input_action = [None] * self.num_agents
        self.msg_down = [None] * self.num_agents
        self.msg_up = [None] * self.num_agents
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        self.parents = parents
        self.num_agents = sum([len(x) for x in parents])
        self.msg_down = [None] * self.num_agents
        self.msg_up = [None] * self.num_agents
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents