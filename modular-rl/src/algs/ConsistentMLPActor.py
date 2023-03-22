import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConsistentMLPPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_action,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super(ConsistentMLPPolicy, self).__init__()
        self.num_agents = 1
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.max_action = max_action
        self.batch_size = batch_size
        self.max_children = max_children
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.monolithic_max_agent = args.monolithic_max_agent
        self.limb_names = args.limb_names
        self.limb_num = dict()
        for env_name in self.limb_names:
            self.limb_num[env_name] = len(self.limb_names[env_name])
        if 'humanoid' in list(self.limb_names.keys())[0]:
            full_limbs = self.limb_names['humanoid_2d_9_full']
        self.limb_idx = dict()
        for env_name in self.limb_names:
            self.limb_idx[env_name] = []
            for limb in self.limb_names[env_name]:
                idx = full_limbs.index(limb)
                self.limb_idx[env_name].append(idx)
            # print (self.limb_names[env_name])
            # print (self.limb_idx[env_name])

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim * self.monolithic_max_agent, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim * self.monolithic_max_agent),
            nn.Tanh(),
        ).to(device)

    def forward(self, state, synergy, mode="train", env_name=None):
        self.clear_buffer()
        batch_size = state.shape[0]
        # self.input_state = state.reshape(batch_size, -1)
        # inpt = F.pad(self.input_state,
        #              pad=[0, self.state_dim * (self.monolithic_max_agent - self.num_agents)],
        #              value=0)
        inpt = torch.zeros(batch_size, self.monolithic_max_agent, self.state_dim, device=device)
        inpt[:, self.limb_idx[env_name], :] = state.reshape(batch_size, -1, self.state_dim)
        inpt = inpt.reshape(batch_size, -1)
        # self.action = self.actor(inpt)[:,:self.num_agents*self.action_dim]
        self.action = self.actor(inpt)
        self.action = self.action.reshape(batch_size, self.monolithic_max_agent, -1)
        self.action = self.action[:, self.limb_idx[env_name], :].reshape(batch_size, -1)
        self.action = self.max_action * self.action

        return torch.squeeze(self.action)


    def change_morphology(self, parents):
        self.parents = parents
        self.num_agents = sum([len(x) for x in parents])
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents

    def clear_buffer(self):
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

