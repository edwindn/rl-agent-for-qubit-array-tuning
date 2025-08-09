import collections
from functools import partial as bind

import elements
import embodied
import numpy as np
import wandb
import json
import time

def get_global_rollout_counter(counter_file="/tmp/qarray_global_rollout_counter.json"):
  with open(counter_file, 'r') as f:
      data = json.load(f)
      now = time.time()
      start_time = data.get("start_time", now)
      elapsed = now - start_time
      return data.get("total_rollouts", 0), elapsed

def train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  print(f"Initialising train with {args.envs} environments")

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  class ManualRewardTracker:
      def __init__(self):
          self.rewards = {}  # worker_id -> accumulated_reward
      
      def reset(self, worker):
          self.rewards[worker] = 0.0
      
      def add(self, worker, reward):
          if worker not in self.rewards:
              self.rewards[worker] = 0.0
          self.rewards[worker] += reward
      
      def get_total(self, worker):
          return self.rewards.get(worker, 0.0)

  # Create tracker instance
  # manual_tracker = ManualRewardTracker()

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    
    # Debug episode state
    if tran['is_first']:
        episode.reset()
        # print(f"DEBUG EPISODE START: Worker {worker} starting new episode")
    
    # Track episode progress
    if not hasattr(logfn, 'step_counts'):
        logfn.step_counts = {}
    if worker not in logfn.step_counts:
        logfn.step_counts[worker] = 0
    
    if tran['is_first']:
        logfn.step_counts[worker] = 0
    else:
        logfn.step_counts[worker] += 1
    
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
        # print(f"EPISODE COMPLETED: Worker {worker} after {logfn.step_counts[worker]} steps")
        
        result = episode.result()
        episode_score = result.pop('score')
        episode_length = result.pop('length')
        
        print(f"Worker {worker} episode score={episode_score}, length={episode_length}")
        
        if args.log_to_wandb:
          if worker == 0:
            rollouts, elapsed = get_global_rollout_counter()
            wandb.log({
              "score": episode_score,
              "length": episode_length,
              "rollouts": rollouts,
              "time elapsed": elapsed
            })
          else:
            wandb.log({
              "score": episode_score,
              "length": episode_length,
            })
        
        rew = result.pop('rewards')
        if len(rew) > 1:
          result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
        # Add score back to result for epstats aggregation
        result['score'] = episode_score
        result['length'] = episode_length
        epstats.add(result)
        
        # Reset step counter
        logfn.step_counts[worker] = 0
    else:
        # Debug why episode isn't ending
        if logfn.step_counts[worker] > 200:  # If episode is getting very long
            print(f"DEBUG LONG EPISODE: Worker {worker} step {logfn.step_counts[worker]} - is_last={tran['is_last']}, is_terminal={tran.get('is_terminal', 'N/A')}")
            

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
