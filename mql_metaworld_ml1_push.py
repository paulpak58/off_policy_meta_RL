import click
import metaworld

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

@click.command()
@click.option('--num_epochs', default=50)
@click.option('--num_train_tasks', default=50)
@click.option('--encoder_hidden_size', default=200)
@click.option('--net_size', default=300)
@click.option('--num_steps_per_epoch', default=2000)
@click.option('--num_initial_steps', default=2000)
@click.option('--num_steps_prior', default=375)
@click.option('--num_extra_rl_steps_posterior', default=375)
@click.option('--batch_size', default=64)
@click.option('--embedding_batch_size', default=64)
@click.option('--embedding_mini_batch_size', default=64)
@wrap_experiment
def mql_metaworld_ml1_push(ctxt=None,
                             seed=1,
                             num_epochs=1000,
                             num_train_tasks=50,
                             latent_size=7,
                             encoder_hidden_size=200,
                             net_size=300,
                             meta_batch_size=16,
                             num_steps_per_epoch=4000,
                             num_initial_steps=4000,
                             num_tasks_sample=15,
                             num_steps_prior=750,
                             num_extra_rl_steps_posterior=750,
                             batch_size=256,
                             embedding_batch_size=64,
                             embedding_mini_batch_size=64,
                             reward_scale=10.,
                             use_gpu=True):
    """Train Meta-Q-Learning with ML1 environments.
    Args:
        ctxt (garage.experiment.ExperimentContext): experiment config used by
            Trainer to create snapshotter
        seed (int): seed to produce determinism
        num_epochs (int): num training epochs.
        num_train_tasks (int): num tasks for training.
        latent_size (int): size latent context vector.
        encoder_hidden_size (int): output dim of dense layer of the context encoder
        net_size (int): output dim of a dense layer of Q-function and value function.
        meta_batch_size (int): meta batch size.
        num_steps_per_epoch (int): num iterations per epoch.
        num_initial_steps (int): num transitions obtained per task before training.
        num_tasks_sample (int): num random tasks to obtain data for each iteration.
        num_steps_prior (int): num transitions to obtain per task with z ~ prior.
        num_extra_rl_steps_posterior (int): num additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int):num transitions in RL batch.
        embedding_batch_size (int): num transitions in context batch.
        embedding_mini_batch_size (int): num transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.
    """
    set_seed(seed)
    encoder_hidden_sizes = ?

    # create meta-learning environment and sample tasks
    ml1 = metaworld.ML1('push-v2')
    train_env = MetaWorldSetTaskEnv(ml1,'train')
    env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env))
    env = env_sampler.sample(num_train_tasks)
    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env))
    trainer = Trainer(ctxt)

    # Network Instantiation
    augmented_env = #TODO PEARL.augment_env_spec(env[0(), latent_size)
    q_func = #TODO ContinuousMLPQFunction(env_spec=augmented_env,hidden_sizes=[net_size,net_size,net_size])
    vfunc_env = #TODO PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = #TODO ContinuousMLPQFunction(env_spec=vfunc_env,hidden_sizes=[net_size,net_size,net_size])
    inner_policy = TanhGaussianMLPPolicy(env_spec=augmented_env,hidden_sizes=[net_size,net_size,net_size])
    sampler = LocalSampler(agents=None,
                           env=env[0](),
                           max_episode_length=env[0]().spec.max_episode_length,
                           n_workers=1,
                           worker_class=PEARLWorker)

    mql = Meta_Q_Learning(...)
    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        mql.to()
    trainer.setup(algo=mql, env=env[0]())
    trainer.train(n_epochs=num_epochs, batch_size=batch_size)


