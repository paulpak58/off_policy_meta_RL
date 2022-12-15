'''
Basic Algorithm Trainer and Sampler for garage
Adopted from docs: https://garage.readthedocs.io/en/latest/user/implement_algo.html
'''
import torch
import numpy as np
from garage import wrap_experiment
from garage.envs import PointEnv,GymEnv
from garage.trainer import Trainer
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.policies import GaussianMLPPolicy
from garage.np import discount_cumsum
from garage import log_performance,EpisodeBatch


class MyAlgorithm:
    def train(self,trainer):
        for epoch in trainer.step_epochs():
            print('Epoch: ',epoch)


@wrap_experiment
def debug_my_algorithm(ctxt):
    trainer = Trainer(ctxt)
    env = PointEnv()
    algo =  MyAlgorithm()
    trainer.setup(algo,env)
    trainer.train(n_epochs=3)

class SimpleVPG:
    def __init__(self,env_spec,policy,sampler):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        self.max_episode_length = 200
        self._discount = 0.99
        self._policy_opt = torch.optim.Adam(self.policy.parameters(),lr=1e-3)

    def train(self,trainer):
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(
                epoch,
                EpisodeBatch.from_episode_list(self.env_spec,samples),
                self._discount
            )
            self._train_once(samples)

    def _train_once(self,samples):
        losses = []
        self._policy_opt.zero_grad()
        for path in samples:
            returns_numpy = discount_cumsum(path['rewards'],self._discount)
            returns = torch.Tensor(returns_numpy.copy())
            obs = torch.Tensor(path['observations'])
            actions = torch.Tensor(path['actions'])
            dist = self.policy(obs)[0]
            log_likelihoods = dist.log_prob(actions)
            loss = (-log_likelihoods*returns).mean()
            loss.backward()
            losses.append(loss.item())
        self._policy_opt.step()
        return np.mean(losses)

@wrap_experiment
def debug_VPG(ctxt):
    set_seed(100)
    trainer = Trainer(ctxt)
    env = PointEnv()
    policy = GaussianMLPPolicy(env.spec)
    sampler = LocalSampler(agents=policy,envs=env,max_episode_length=200)
    algo = SimpleVPG(env.spec,policy,sampler)
    trainer.setup(algo,env)
    trainer.train(n_epochs=500,batch_size=4000)

@wrap_experiment
def lunar_vpg(ctxt=None):
    set_seed(100)
    trainer = Trainer(ctxt)
    env = GymEnv('LunarLanderContinuous-v2')
    policy = GaussianMLPPolicy(env.spec)
    sampler = LocalSampler(agents=policy,envs=env,max_episode_length=200)
    algo = SimpleVPG(env.spec,policy,sampler)
    trainer.setup(algo,env)
    trainer.train(n_epochs=500,batch_size=4000,plot=True)

debug_my_algorithm()
print('Finished')
lunar_vpg()

