import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from env import Env
from network import AttentionNet
from arguments import arg


class Worker:
    def __init__(self, meta_id, local_net, global_step, budget_size, graph_size=arg.graph_size[0], history_size=arg.history_size[0],
                 target_size=arg.target_size[0], device='cuda', greedy=False, save_image=False):
        self.meta_id = meta_id
        self.device = device
        self.greedy = greedy
        self.global_step = global_step
        self.save_image = save_image
        self.graph_size = graph_size
        self.history_size = history_size
        self.env = Env(graph_size=self.graph_size, k_size=arg.k_size, budget_size=budget_size, target_size=target_size)
        self.local_net = local_net
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg.history_stride, stride=arg.history_stride, ceil_mode=True)
        self.episode_buffer_keys = ['history', 'edge', 'dist', 'dt', 'nodeidx', 'logp', 'action', 'value', 'temporalmask',
                                    'spatiomask', 'spatiope', 'done', 'reward', 'advantage', 'return']

    def reset_env_input(self):
        node_coords, graph, node_feature, budget = self.env.reset()  # node_feature: Array (node, (target x feature))
        node_inputs = np.concatenate((node_coords, node_feature), axis=1)
        node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node, 2+targetxfeature)
        node_history = node_inputs.repeat(self.history_size, 1, 1)  # (history, node, 2+targetxfeature)
        history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)  # (1, hpool, n, 2+targetxfeature)

        edge_inputs = [list(map(int, node)) for node in graph.values()]
        spatio_pos_encoding = self.graph_pos_encoding(edge_inputs)
        spatio_pos_encoding = torch.from_numpy(spatio_pos_encoding).float().unsqueeze(0).to(self.device)  # (1, node, 32)
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, node, k)

        dt_history = torch.zeros((1, self.history_size, 1)).to(self.device)  # (1, history, 1)
        dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)  # (1, hpool, 1)
        dist_inputs = self.calc_distance_to_nodes(current_idx=self.env.current_node_index)
        dist_inputs[dist_inputs > 1.5] = 1.5
        dist_inputs = torch.Tensor(dist_inputs).unsqueeze(0).to(self.device)  # (1, node, 1)

        current_index = torch.tensor([[[self.env.current_node_index]]]).to(self.device)

        spatio_mask = torch.zeros((1, self.graph_size + 1, arg.k_size), dtype=torch.bool).to(self.device)
        temporal_mask = torch.tensor([1])
        return node_coords, node_history, history_pool_inputs, edge_inputs, dist_inputs, dt_history, dt_pool_inputs, \
               current_index, spatio_pos_encoding, temporal_mask, spatio_mask

    def run_episode(self, episode_number):
        perf_metrics = dict()
        episode_buffer = {k: [] for k in self.episode_buffer_keys}
        node_coords, node_history, history_pool_inputs, edge_inputs, dist_inputs, dt_history, dt_pool_inputs, \
            current_index, spatio_pos_encoding, temporal_mask, spatio_mask = self.reset_env_input()
        route = [current_index.item()]
        rmse_list = [self.env.RMSE]
        unc_list = [self.env.unc_list]
        jsd_list = [self.env.JS_list]
        kld_list = [self.env.KL_list]
        unc_stddev_list = [np.std(self.env.unc_list)]
        jsd_stddev_list = [np.std(self.env.JS_list)]
        budget_list = [0]

        for step in range(arg.episode_steps):
            if self.save_image:
                self.env.plot(route, self.global_step, step, arg.gifs_path, budget_list,
                              [0] + [r.item() for r in episode_buffer['reward']], jsd_list)

            with torch.no_grad():
                logp_list, value = self.local_net(history_pool_inputs, edge_inputs, dist_inputs, dt_pool_inputs,
                                                  current_index, spatio_pos_encoding, temporal_mask, spatio_mask)
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            logp = torch.gather(logp_list, 1, action_index.unsqueeze(0))
            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            reward, done, node_feature, remain_budget, _ = self.env.step(next_node_index.item(), self.global_step)

            episode_buffer['history'] += history_pool_inputs
            episode_buffer['edge'] += edge_inputs
            episode_buffer['dist'] += dist_inputs
            episode_buffer['dt'] += dt_pool_inputs
            episode_buffer['nodeidx'] += current_index
            episode_buffer['logp'] += logp.unsqueeze(0)
            episode_buffer['action'] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer['value'] += value
            episode_buffer['temporalmask'] += temporal_mask
            episode_buffer['spatiomask'] += spatio_mask
            episode_buffer['spatiope'] += spatio_pos_encoding
            episode_buffer['reward'] += torch.Tensor([[[reward]]]).to(self.device)
            episode_buffer['done'] += [done]

            route += [next_node_index.item()]
            rmse_list += [self.env.RMSE]
            unc_list += [self.env.unc_list]
            jsd_list += [self.env.JS_list]
            kld_list += [self.env.KL_list]
            unc_stddev_list += [np.std(self.env.unc_list)]
            jsd_stddev_list += [np.std(self.env.JS_list)]
            budget_list += [self.env.budget_init - remain_budget]

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_inputs = np.concatenate((node_coords, node_feature), axis=1)
            node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)
            node_history = torch.cat((node_history, node_inputs.clone()), dim=0)[-self.history_size:, :, :]
            history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
            dt_history += (budget_list[-1] - budget_list[-2]) / (1.993 * 3)  # 1% unc with timescale
            dt_history = torch.cat((dt_history, torch.tensor([[[0]]], device=self.device)), dim=1)[:, -self.history_size:, :]
            dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)
            dist_inputs = self.calc_distance_to_nodes(current_idx=current_index.item())
            dist_inputs[dist_inputs > 1.5] = 1.5
            dist_inputs = torch.Tensor(dist_inputs).unsqueeze(0).to(self.device)

            # mask
            spatio_mask = torch.zeros((1, self.graph_size + 1, arg.k_size), dtype=torch.bool).to(self.device)
            temporal_mask = torch.tensor([(len(route) - 1) // arg.history_stride + 1])

            if done:
                # save gif
                if self.save_image:
                    self.env.plot(route, self.global_step, step + 1, arg.gifs_path, budget_list,
                                  [0] + [r.item() for r in episode_buffer['reward']], jsd_list)
                    self.make_gif(arg.gifs_path, episode_number)
                    self.save_image = False
                node_coords, node_history, history_pool_inputs, edge_inputs, dist_inputs, dt_history, dt_pool_inputs, \
                    current_index, spatio_pos_encoding, temporal_mask, spatio_mask = self.reset_env_input()
                route = [current_index.item()]
                rmse_list = [self.env.RMSE]
                jsd_list = [self.env.JS_list]
                kld_list = [self.env.KL_list]
                budget_list = [0]

        # save gif
        if self.save_image:
            self.env.plot(route, self.global_step, step + 1, arg.gifs_path, budget_list,
                          [0] + [r.item() for r in episode_buffer['reward']], jsd_list)
            self.make_gif(arg.gifs_path, episode_number)
            self.save_image = False
        n_visit = list(map(len, self.env.visit_t))
        gap_visit = list(map(np.diff, self.env.visit_t))
        perf_metrics['avgnvisit'] = np.mean(n_visit)
        perf_metrics['stdnvisit'] = np.std(n_visit)
        perf_metrics['avggapvisit'] = np.mean(list(map(np.mean, gap_visit))) if min(n_visit) > 1 else np.nan
        perf_metrics['stdgapvisit'] = np.std(list(map(np.mean, gap_visit))) if min(n_visit) > 1 else np.nan
        perf_metrics['avgrmse'] = np.mean(rmse_list)
        perf_metrics['avgunc'] = np.mean(unc_list)
        perf_metrics['avgjsd'] = np.mean(jsd_list)
        perf_metrics['avgkld'] = np.mean(kld_list)
        perf_metrics['stdunc'] = np.mean(unc_stddev_list)
        perf_metrics['stdjsd'] = np.mean(jsd_stddev_list)
        perf_metrics['f1'] = self.env.gp_wrapper.eval_avg_F1(self.env.ground_truth, self.env.curr_t)
        perf_metrics['mi'] = self.env.gp_wrapper.eval_avg_MI(self.env.curr_t)
        perf_metrics['covtr'] = self.env.cov_trace
        perf_metrics['js'] = self.env.JS
        perf_metrics['rmse'] = self.env.RMSE
        perf_metrics['scalex'] = 0.1  # self.env.GPs.gp.kernel_.length_scale[0]
        perf_metrics['scalet'] = 3  # scale_t
        print('\033[92m' + 'meta{:02}:'.format(self.meta_id) + '\033[0m',
              'episode {} done at {} steps, avg JS {:.4g}'.format(episode_number, step, perf_metrics['avgjsd']))

        with torch.no_grad():
            if not done:
                _, next_value = self.local_net(history_pool_inputs, edge_inputs, dist_inputs, dt_pool_inputs,
                                               current_index, spatio_pos_encoding, temporal_mask, spatio_mask)  # bootstrap
                next_value = next_value.item()
            else:
                next_value = 0

            # GAE
            lastgaelam = 0
            for i in reversed(range(arg.episode_steps)):
                if i == arg.episode_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalue = next_value
                else:
                    nextnonterminal = 1.0 - episode_buffer['done'][i + 1]
                    nextvalue = episode_buffer['value'][i + 1].item()
                delta = episode_buffer['reward'][i].item() + arg.gamma * nextvalue * nextnonterminal - episode_buffer['value'][i].item()
                lastgaelam = delta + arg.gamma * arg.gae_lambda * nextnonterminal * lastgaelam
                episode_buffer['advantage'].insert(0, torch.Tensor([[lastgaelam]]).to(self.device))
            episode_buffer['return'] = [adv + val for adv, val in zip(episode_buffer['advantage'], episode_buffer['value'])]

        return episode_buffer, perf_metrics

    def graph_pos_encoding(self, edge_inputs):
        A_matrix = np.zeros((self.graph_size + 1, self.graph_size + 1))
        D_matrix = np.zeros((self.graph_size + 1, self.graph_size + 1))
        for i in range(self.graph_size + 1):
            for j in range(self.graph_size + 1):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.graph_size + 1):
            D_matrix[i][i] = 1 / np.sqrt(len(edge_inputs[i]) - 1)
        L = np.eye(self.graph_size + 1) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:, 1:32 + 1]
        return eigen_vector

    def calc_distance_to_nodes(self, current_idx):
        all_dist = []
        current_coord = self.env.node_coords[current_idx]
        for i, point_coord in enumerate(self.env.node_coords):
            dist_current_to_point = self.env.graph_ctrl.calc_distance(current_coord, point_coord)
            all_dist.append(dist_current_to_point)
        return np.asarray(all_dist).reshape(-1, 1)

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.mp4'.format(path, n, self.env.cov_trace), fps=5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')
        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


if __name__ == '__main__':
    save_img = False
    if save_img:
        if not os.path.exists(arg.gifs_path):
            os.makedirs(arg.gifs_path)
    device = torch.device('cuda')
    localNetwork = AttentionNet(arg.embedding_dim).cuda()
    worker = Worker(0, localNetwork, 100000, budget_size=30, graph_size=200, history_size=50, target_size=3, save_image=save_img)
    worker.run_episode(0)
