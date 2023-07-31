import copy
import os
import imageio
import numpy as np
import time
import torch
import torch.nn.functional as F
from env import Env
from network import AttentionNet
from arguments import arg_eval


class WorkerEval:
    def __init__(self, meta_id, local_net, global_step, device='cuda', greedy=True, save_image=False, config=None):
        self.device = device
        self.greedy = greedy
        self.meta_id = meta_id
        self.global_step = global_step
        self.save_image = save_image
        if config:
            arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config
        print(f'#node{arg_eval.graph_size}, #history{arg_eval.history_size}, #tgt{arg_eval.target_size}, speed{arg_eval.target_speed}')
        self.env_speed = arg_eval.target_speed
        self.env = Env(graph_size=arg_eval.graph_size, k_size=arg_eval.k_size, budget_size=arg_eval.budget_size,
                       target_size=arg_eval.target_size)
        self.local_net = local_net
        self.avgpool = torch.nn.AvgPool1d(kernel_size=arg_eval.history_stride, stride=arg_eval.history_stride, ceil_mode=True)
        self.perf_metrics = None
        self.planning_time = 0

    def run_episode(self, curr_eval):
        perf_metrics = dict()
        node_coords, graph, node_feature, budget = self.env.reset(seed=self.global_step)  # node_feature: Array (node, (target x feature))
        node_inputs = np.concatenate((node_coords, node_feature), axis=1)
        node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node, 2+targetxfeature)
        node_history = node_inputs.repeat(arg_eval.history_size, 1, 1)
        history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)

        edge_inputs = [list(map(int, node)) for node in graph.values()]
        spatio_pos_encoding = self.graph_pos_encoding(edge_inputs)
        spatio_pos_encoding = torch.from_numpy(spatio_pos_encoding).float().unsqueeze(0).to(self.device)  # (1, node, 32)
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, node, k)

        dt_history = torch.zeros((1, arg_eval.history_size, 1)).to(self.device)  # (1, history, 1)
        dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)  # (1, hpool, 1)
        dist_inputs = self.calc_distance_to_nodes(current_idx=self.env.current_node_index)
        dist_inputs = torch.Tensor(dist_inputs).unsqueeze(0).to(self.device)  # (1, node, 1)

        current_index = torch.tensor([[[self.env.current_node_index]]]).to(self.device)
        spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.int64).to(self.device)
        temporal_mask = torch.tensor([1])

        route = [current_index.item()]
        rmse_list = [self.env.RMSE]
        jsd_list = [self.env.JS]
        jsd_all_list = [self.env.JS_list]
        unc_list = [self.env.unc]
        unc_all_list = [self.env.unc_list]
        unc_stddev_list = [np.std(self.env.unc_list)]
        jsd_stddev_list = [np.std(self.env.JS_list)]
        d_to_target_list = [self.env.d_to_target]
        budget_list = [0]
        budget_at_node = [0]

        for step in range(1024):
            if self.save_image:
                self.env.plot(route, self.global_step, step, arg_eval.result_path, budget_list, [0], d_to_target_list)
            time_start = time.time()
            with torch.no_grad():
                logp_list, value = self.local_net(history_pool_inputs, edge_inputs, dist_inputs, dt_pool_inputs,
                                                  current_index, spatio_pos_encoding, temporal_mask, spatio_mask)
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            reward, done, node_feature, remain_budget, metrics = self.env.step(next_node_index.item(), eval_speed=self.env_speed)
            time_end = time.time()
            self.planning_time += time_end - time_start

            route += [next_node_index.item()]
            rmse_list += metrics['rmse']
            jsd_list += metrics['jsd']
            jsd_all_list += metrics['jsdall']
            unc_list += metrics['unc']
            unc_all_list += metrics['uncall']
            unc_stddev_list += metrics['uncstd']
            jsd_stddev_list += metrics['jsdstd']
            d_to_target_list += metrics['dtotarget']
            budget_list += metrics['budget']
            budget_at_node += [budget - remain_budget]

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_inputs = np.concatenate((node_coords, node_feature), axis=1)
            node_inputs = torch.Tensor(node_inputs).unsqueeze(0).to(self.device)
            node_history = torch.cat((node_history, node_inputs.clone()), dim=0)[-arg_eval.history_size:, :, :]
            history_pool_inputs = self.avgpool(node_history.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0)
            dt_history += (budget_at_node[-1] - budget_at_node[-2]) / (1.993 * 3)  # 1% unc with timescale
            dt_history = torch.cat((dt_history, torch.tensor([[[0]]], device=self.device)), dim=1)[:, -arg_eval.history_size:, :]
            dt_pool_inputs = self.avgpool(dt_history.permute(0, 2, 1)).permute(0, 2, 1)
            dist_inputs = self.calc_distance_to_nodes(current_idx=current_index.item())
            dist_inputs = torch.Tensor(dist_inputs).unsqueeze(0).to(self.device)

            spatio_mask = torch.zeros((1, arg_eval.graph_size + 1, arg_eval.k_size), dtype=torch.int64).to(self.device)
            temporal_mask = torch.tensor([(len(route) - 1) // arg_eval.history_stride + 1])

            if done:
                if self.save_image:
                    self.env.plot(route, self.global_step, step + 1, arg_eval.result_path, budget_list, [0], d_to_target_list)
                    self.make_gif(arg_eval.result_path, self.global_step)
                n_visit = list(map(len, self.env.visit_t))
                gap_visit = list(map(np.diff, self.env.visit_t))
                perf_metrics['minnvisit'] = np.min(n_visit)
                perf_metrics['avgnvisit'] = np.mean(n_visit)
                perf_metrics['stdnvisit'] = np.std(n_visit)
                perf_metrics['avggapvisit'] = np.mean(list(map(np.mean, gap_visit))) if min(n_visit) > 1 else np.nan
                perf_metrics['stdgapvisit'] = np.std(list(map(np.mean, gap_visit))) if min(n_visit) > 1 else np.nan
                perf_metrics['avgrmse'] = np.mean(rmse_list)
                perf_metrics['avgjsd'] = np.mean(jsd_list)
                perf_metrics['avgunc'] = np.mean(unc_list)
                perf_metrics['stdunc'] = np.mean(unc_stddev_list)
                perf_metrics['stdjsd'] = np.mean(jsd_stddev_list)
                perf_metrics['budget_list'] = budget_list
                perf_metrics['rmse_list'] = rmse_list
                perf_metrics['jsd_list'] = jsd_list
                perf_metrics['unc_list'] = unc_list
                print('\033[92m' + 'meta{:02}:'.format(self.meta_id) + '\033[0m',
                      'episode {} done at {} steps, avg JS {:.4g}'.format(curr_eval, step, perf_metrics['avgjsd']))
                break
        print(f'episode {self.global_step} planning time: {self.planning_time/len(route):.3f}/step')
        return perf_metrics

    @staticmethod
    def graph_pos_encoding(edge_inputs):
        A_matrix = np.zeros((arg_eval.graph_size + 1, arg_eval.graph_size + 1))
        D_matrix = np.zeros((arg_eval.graph_size + 1, arg_eval.graph_size + 1))
        for i in range(arg_eval.graph_size + 1):
            for j in range(arg_eval.graph_size + 1):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(arg_eval.graph_size + 1):
            D_matrix[i][i] = 1 / np.sqrt(len(edge_inputs[i]) - 1)
        L = np.eye(arg_eval.graph_size + 1) - np.matmul(D_matrix, A_matrix, D_matrix)
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
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


if __name__ == '__main__':
    save_img = False
    if save_img:
        if not os.path.exists(arg_eval.gifs_path):
            os.makedirs(arg_eval.gifs_path)
    device = torch.device('cuda')
    localNetwork = AttentionNet(arg_eval.embedding_dim).cuda()
    checkpoint = torch.load(f'../{arg_eval.model_path}/checkpoint.pth')
    localNetwork.load_state_dict(checkpoint['model'])
    worker = WorkerEval(0, localNetwork, 2, save_image=save_img)
    worker.run_episode(0)