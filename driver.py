import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import wandb
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from network import AttentionNet
from runner import Runner
from arguments import arg


class Logger:
    def __init__(self):
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.cuda_devices = str(arg.cuda_devices)[1:-1]
        self.writer = SummaryWriter(arg.train_path) if arg.save_files else None
        self.episode_buffer_keys = ['history', 'edge', 'dist', 'dt',  'nodeidx', 'logp', 'action', 'value',
                                    'temporalmask', 'spatiomask', 'spatiope', 'done', 'reward', 'advantage', 'return']
        self.metric_names = ['avgnvisit', 'stdnvisit', 'avggapvisit', 'stdgapvisit', 'avgrmse', 'avgunc', 'avgjsd',
                             'avgkld', 'stdunc', 'stdjsd', 'covtr', 'f1', 'mi', 'js', 'rmse', 'scalex', 'scalet']
        np.random.seed(0)
        print('=== Welcome to STAMP! ===\n'
              f'Initializing : {arg.run_name}\n'
              f'Minibatch size : {arg.minibatch_size}, Buffer size : {arg.buffer_size}')
        if self.cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_devices
            print(f'cuda devices : {self.cuda_devices} on', torch.cuda.get_device_name())
        ray.init()
        if arg.use_wandb:
            wandb.init(project=arg.project_name, name=arg.run_name, entity='your_entity', config=vars(arg),
                       notes=arg.wandb_notes, resume='allow', id=arg.wandb_id)
        if arg.save_files: os.makedirs(arg.model_path, exist_ok=True)
        if arg.save_files: os.makedirs(arg.gifs_path, exist_ok=True)

    def set(self, net, optimizer, lr_scheduler):
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def write_to_board(self, data, curr_episode):
        data = np.array(data)
        data = list(np.nanmean(data, axis=0))
        reward, value, p_loss, v_loss, entropy, grad_norm, returns, clipfrac, approx_kl, \
        avg_nvisit, std_nvisit, avg_visitgap, std_visitgap, avg_RMSE, avg_unc, avg_JSD, avg_KLD, std_unc, std_JSD, \
        cov_tr, F1, MI, JSD, RMSE, sx, st = data
        metrics = {'Loss/Learning Rate': self.lr_scheduler.get_last_lr()[0],
                   'Loss/Value': value,
                   'Loss/Policy Loss': p_loss,
                   'Loss/Value Loss': v_loss,
                   'Loss/Entropy': entropy,
                   'Loss/Grad Norm': grad_norm,
                   'Loss/Clip Frac': clipfrac,
                   'Loss/Approx Policy KL': approx_kl,
                   'Loss/Reward': reward,
                   'Loss/Return': returns,
                   'Perf/Average Visit Times': avg_nvisit,
                   'Perf/Stddev Visit Times': std_nvisit,
                   'Perf/Average Visit Gap': avg_visitgap,
                   'Perf/Stddev Visit Gap': std_visitgap,
                   'Perf/Average JS Div': avg_JSD,
                   'Perf/Average KL Div': avg_KLD,
                   'Perf/Average RMSE': avg_RMSE,
                   'Perf/Average Unc': avg_unc,
                   'Perf/Stddev Unc': std_unc,
                   'Perf/Stddev JS Div': std_JSD,
                   'Perf/JS Div': JSD,
                   'Perf/RMSE': RMSE,
                   'Perf/F1 Score': F1,
                   'GP/Mutual Info': MI,
                   'GP/Cov Trace': cov_tr,
                   'GP/Length Scale x': sx,
                   'GP/Length Scale t': st
                   }
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)
        if arg.use_wandb:
            wandb.log(metrics, step=curr_episode)

    def load_saved_model(self):
        print('Loading model :', arg.run_name)
        checkpoint = torch.load(arg.model_path + '/checkpoint.pth')
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("Current episode set to :", curr_episode)
        print('Learning rate :', self.optimizer.state_dict()['param_groups'][0]['lr'])
        return curr_episode

    def save_model(self, curr_episode):
        print('Saving model', end='\n')
        checkpoint = {"model": self.net.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "episode": curr_episode,
                      "lr_decay": self.lr_scheduler.state_dict()}
        path_checkpoint = "./" + arg.model_path + "/checkpoint.pth"
        torch.save(checkpoint, path_checkpoint)


def main():
    logger = Logger()
    device = torch.device('cuda') if arg.use_gpu_driver else torch.device('cpu')
    local_device = torch.device('cuda') if arg.use_gpu_runner else torch.device('cpu')
    global_network = AttentionNet(arg.embedding_dim).to(device)
    # global_network.share_memory()
    global_optimizer = optim.Adam(global_network.parameters(), lr=arg.lr)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=arg.lr_decay_step, gamma=0.96)
    logger.set(global_network, global_optimizer, lr_decay)

    curr_episode = 0
    training_data = []

    if arg.load_model:
        curr_episode = logger.load_saved_model()

    # launch meta agents
    meta_runners = [Runner.remote(i) for i in range(arg.num_meta)]

    # launch the first job on each runner
    if arg.use_wandb: wandb.watch(global_network, log_freq=500, log_graph=True)
    dp_global_network = nn.DataParallel(global_network)

    try:
        while True:
            meta_jobs = []
            buffer = {k: [] for k in logger.episode_buffer_keys}
            buffer_idxs = np.arange(arg.buffer_size)
            budget_size = np.random.uniform(*arg.budget_size)
            graph_size = np.random.randint(*arg.graph_size)
            history_size = np.random.randint(*arg.history_size)
            target_size = np.random.randint(*arg.target_size)
            # get global weights
            if device != local_device:
                weights = global_network.to(local_device).state_dict()
                global_network.to(device)
            else:
                weights = global_network.state_dict()
            weights_id = ray.put(weights)

            for i, meta_agent in enumerate(meta_runners):
                meta_jobs.append(meta_agent.job.remote(weights_id, curr_episode, budget_size, graph_size, history_size,
                                                       target_size))
                curr_episode += 1

            done_id, meta_jobs = ray.wait(meta_jobs, num_returns=arg.num_meta)
            done_jobs = ray.get(done_id)
            # random.shuffle(done_jobs)
            perf_metrics = {}
            for n in logger.metric_names:
                perf_metrics[n] = []
            for job in done_jobs:
                job_results, metrics = job
                for k in job_results.keys():
                    buffer[k] += job_results[k]
                for n in logger.metric_names:
                    perf_metrics[n].append(metrics[n])

            b_history_inputs = torch.stack(buffer['history'], dim=0)
            b_edge_inputs = torch.stack(buffer['edge'], dim=0)
            b_dist_inputs = torch.stack(buffer['dist'], dim=0)
            b_dt_inputs = torch.stack(buffer['dt'], dim=0)
            b_current_inputs = torch.stack(buffer['nodeidx'], dim=0)
            b_logp = torch.stack(buffer['logp'], dim=0)
            b_action = torch.stack(buffer['action'], dim=0)
            b_value = torch.stack(buffer['value'], dim=0)
            b_reward = torch.stack(buffer['reward'], dim=0)
            b_return = torch.stack(buffer['return'], dim=0)
            b_advantage = torch.stack(buffer['advantage'], dim=0)
            b_temporal_mask = torch.stack(buffer['temporalmask'])
            b_spatio_mask = torch.stack(buffer['spatiomask'])
            b_pos_encoding = torch.stack(buffer['spatiope'])

            scaler = GradScaler()
            for epoch in range(arg.update_epochs):
                np.random.shuffle(buffer_idxs)
                for start in range(0, arg.buffer_size, arg.minibatch_size):
                    end = start + arg.minibatch_size
                    mb_idxs = buffer_idxs[start:end]
                    mb_old_logp = b_logp[mb_idxs].to(device)
                    mb_history_inputs = b_history_inputs[mb_idxs].to(device)
                    mb_edge_inputs = b_edge_inputs[mb_idxs].to(device)
                    mb_dist_inputs = b_dist_inputs[mb_idxs].to(device)
                    mb_dt_inputs = b_dt_inputs[mb_idxs].to(device)
                    mb_current_inputs = b_current_inputs[mb_idxs].to(device)
                    mb_action = b_action[mb_idxs].to(device)
                    mb_return = b_return[mb_idxs].to(device)
                    mb_advantage = b_advantage[mb_idxs].to(device)
                    mb_temporal_mask = b_temporal_mask[mb_idxs].to(device)
                    mb_spatio_mask = b_spatio_mask[mb_idxs].to(device)
                    mb_pos_encoding = b_pos_encoding[mb_idxs].to(device)

                    with autocast():
                        logp_list, value = dp_global_network(mb_history_inputs, mb_edge_inputs, mb_dist_inputs, mb_dt_inputs,
                                                             mb_current_inputs, mb_pos_encoding, mb_temporal_mask, mb_spatio_mask)
                        logp = torch.gather(logp_list, 1, mb_action.squeeze(1)).unsqueeze(1)
                        logratio = logp - mb_old_logp.detach()
                        ratio = logratio.exp()
                        surr1 = mb_advantage.detach() * ratio
                        surr2 = mb_advantage.detach() * ratio.clamp(1-0.2, 1+0.2)

                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = nn.MSELoss()(value, mb_return).mean()
                        entropy = -(logp_list * logp_list.exp()).sum(dim=-1).mean()
                        loss = policy_loss + 0.2 * value_loss - 0.0 * entropy

                    global_optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(global_optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=5, norm_type=2)
                    scaler.step(global_optimizer)
                    scaler.update()
            lr_decay.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > 0.2).float().mean()
                approx_kl = ((ratio - 1) - logratio).mean()

            perf_data = []
            for n in logger.metric_names:
                perf_data.append(np.nanmean(perf_metrics[n]))
            data = [b_reward.mean().item(), b_value.mean().item(), policy_loss.item(), value_loss.item(), entropy.item(),
                    grad_norm.item(), b_return.mean().item(), clip_frac.item(), approx_kl.item(), *perf_data]
            training_data.append(data)
            if len(training_data) >= arg.summary_window and arg.save_files:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            if curr_episode % 64 == 0 and arg.save_files:
                logger.save_model(curr_episode)


    except KeyboardInterrupt:
        print('User interrupt, abort remotes...')
        if arg.use_wandb: wandb.finish(quiet=True)
        for runner in meta_runners:
            ray.kill(runner)


if __name__ == "__main__":
    main()
