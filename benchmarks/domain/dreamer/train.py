"""
DreamerV3 training script for quantum dot array tuning.

Usage:
    python train.py --num_dots 2 --steps 100000 --logdir /tmp/dreamer_qd
    python train.py --num_dots 4 --steps 500000 --configs size12m
"""

import argparse
import importlib
import os
import pathlib
import sys
from functools import partial as bind

# Parse --gpu early before JAX imports (must be done first)
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
_gpu_args, _ = _gpu_parser.parse_known_args()

# Restrict JAX to single GPU (must be set before importing JAX)
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_args.gpu)

# Setup paths
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder))
sys.path.insert(1, str(folder.parent))
sys.path.insert(2, str(folder.parent.parent / 'src'))

# JAX configuration (before importing JAX)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

import ninjax_patch  # Patch ninjax 3.6.2 debug print (see ninjax_patch.py)

from wrapper import make_dreamer_env

# reset distance logging race file
race_file = folder / "distance_logging.lock"
try:
    race_file.unlink()
except FileNotFoundError:
    pass


def main(argv=None):
    from agent import Agent
    [elements.print(line) for line in Agent.banner]

    # Parse command line for custom arguments first
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_dots', type=int, default=2, help='Number of quantum dots')
    parser.add_argument('--use_barriers', type=bool, default=True, help='Control barriers')
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--steps', type=int, default=None, help='Total training steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device (already applied)')
    custom_args, remaining = parser.parse_known_args(argv)

    # Load configs
    configs = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    parsed, other = elements.Flags(configs=['defaults', 'size12m']).parse_known(remaining)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)

    # Apply custom arguments
    if custom_args.steps:
        config = config.update({'run': {'steps': custom_args.steps}})
    if custom_args.seed is not None:
        config = config.update(seed=custom_args.seed)

    # Store env config for make_env
    config = config.update({
        'env': {
            'quantum': {
                'num_dots': custom_args.num_dots,
                'use_barriers': custom_args.use_barriers,
                'max_steps': custom_args.max_steps,
            }
        }
    })

    config = config.update(logdir=(
        config.logdir.format(timestamp=elements.timestamp())))

    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
    print('Replica:', config.replica, '/', config.replicas)

    logdir = elements.Path(config.logdir)
    print('Logdir:', logdir)
    print('Run script:', config.script)
    print(f'Quantum dots: {custom_args.num_dots}, Barriers: {custom_args.use_barriers}')

    if not config.script.endswith(('_env', '_replay')):
        logdir.mkdir()
        config.save(logdir / 'config.yaml')

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    # Simplified portal setup (no distributed training)
    try:
        import portal
        portal.setup(
            errfile=config.errfile and logdir / 'error',
            clientkw=dict(logging_color='cyan'),
            serverkw=dict(logging_color='cyan'),
            initfns=[init],
            ipv6=config.ipv6,
        )
    except ImportError:
        print("Portal not available, running without distributed support")
        init()

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    if config.script == 'train':
        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)

    elif config.script == 'train_eval':
        embodied.run.train_eval(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_replay, config, 'eval_replay', 'eval'),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)

    elif config.script == 'eval_only':
        embodied.run.eval_only(
            bind(make_agent, config),
            bind(make_env, config),
            bind(make_logger, config),
            args)

    else:
        raise NotImplementedError(config.script)


def make_agent(config):
    from agent import Agent
    env = make_env(config, 0)
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


def make_logger(config):
    step = elements.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
    for output in config.logger.outputs:
        if output == 'jsonl':
            outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
            outputs.append(elements.logger.JSONLOutput(
                logdir, 'scores.jsonl', 'episode/score'))
        elif output == 'tensorboard':
            outputs.append(elements.logger.TensorBoardOutput(
                logdir, config.logger.fps))
        elif output == 'wandb':
            name = '/'.join(logdir.split('/')[-4:])
            outputs.append(elements.logger.WandBOutput(name))
        else:
            print(f"Unknown logger output: {output}")
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, folder, mode='train'):
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity

    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f'{config.replica:05}'
    kwargs = dict(
        length=length, capacity=int(capacity), online=config.replay.online,
        chunksize=config.replay.chunksize, directory=directory)

    if config.replay.fracs.uniform < 1 and mode == 'train':
        assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
            'Gradient scaling for low-precision training can produce invalid loss '
            'outputs that are incompatible with prioritized replay.')
        prio = embodied.replay.selectors.Prioritized(
            seed=config.seed + config.replica,
            initial=config.replay.initial_prio,
            exponent=config.replay.prio_exponent,
            zero_on_sample=config.replay.zero_on_sample,
        )
        recen = embodied.replay.selectors.Recency(
            seed=config.seed + config.replica,
            halflife=config.replay.recen_halflife,
            initial=config.replay.recen_initial,
        )
        kwargs['selector'] = embodied.replay.selectors.Mixture(
            seed=config.seed + config.replica,
            selectors={'prio': prio, 'recen': recen, 'uniform': None},
            fracs=config.replay.fracs,
        )
    return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
    """Create quantum dot environment wrapped for DreamerV3."""
    from embodied.envs import from_gym

    # Get quantum environment config
    qconfig = config.env.get('quantum', {})
    num_dots = qconfig.get('num_dots', 2)
    use_barriers = qconfig.get('use_barriers', True)
    max_steps = qconfig.get('max_steps', 50)

    # Create wrapped environment
    gym_env = make_dreamer_env(
        num_dots=num_dots,
        use_barriers=use_barriers,
        max_steps=max_steps,
        seed=hash((config.seed, index)) % (2 ** 32 - 1)
    )

    env = from_gym.FromGym(gym_env)
    print(f"Created quantum dot environment: {num_dots} dots, barriers={use_barriers}")

    return wrap_env(env, config)


def wrap_env(env, config):
    """Apply standard DreamerV3 wrappers."""
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_stream(config, replay, mode):
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True)
    return stream


if __name__ == '__main__':
    main()
