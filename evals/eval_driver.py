import csv
import os
import ray
import torch
import numpy as np
import time
from network import AttentionNet
from runner import Runner
from eval_worker import WorkerEval
from arguments import arg_eval


def main(config=None):
    if config:
        arg_eval.graph_size, arg_eval.history_size, arg_eval.target_size, arg_eval.target_speed = config
    os.makedirs(arg_eval.result_path, exist_ok=True)
    device = torch.device('cuda') if arg_eval.use_gpu_driver else torch.device('cpu')
    local_device = torch.device('cuda') if arg_eval.use_gpu_runner else torch.device('cpu')
    global_network = AttentionNet(arg_eval.embedding_dim).to(device)
    checkpoint = torch.load(f'../{arg_eval.model_path}/checkpoint.pth')
    global_network.load_state_dict(checkpoint['model'])

    print(f'Loading model: {arg_eval.run_name}...')

    # init meta agents
    meta_runners = [Runner.remote(i) for i in range(arg_eval.num_meta)]
    weights = global_network.to(local_device).state_dict() if device != local_device else global_network.state_dict()
    curr_test = 1
    metric_names = ['avgjsd', 'avgunc', 'minnvisit', 'avgrmse', 'stdunc', 'stdjsd', 'avgnvisit', 'stdnvisit', 'avggapvisit', 'stdgapvisit']
    perf_metrics = {}
    for n in metric_names:
        perf_metrics[n] = []

    eval_num_list = []
    budget_list = []
    rmse_list = []
    jsd_list = []
    unc_list = []
    avgjsd_list = []
    avgunc_list = []
    stdjsd_list = []
    stdunc_list = []
    minvisit_list = []
    avgvisit_list = []
    stdvisit_list = []

    try:
        while True:
            jobList = []
            for i, meta_agent in enumerate(meta_runners):
                jobList.append(meta_agent.job.remote(weights, curr_test, config))
                curr_test += 1
            done_id, jobList = ray.wait(jobList, num_returns=arg_eval.num_meta)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, eval_num = job
                eval_num_list += [eval_num]
                avgjsd_list += [metrics['avgjsd']]
                avgunc_list += [metrics['avgunc']]
                stdjsd_list += [metrics['stdjsd']]
                stdunc_list += [metrics['stdunc']]
                minvisit_list += [metrics['minnvisit']]
                avgvisit_list += [metrics['avgnvisit']]
                stdvisit_list += [metrics['stdnvisit']]
                for n in metric_names:
                    perf_metrics[n].append(metrics[n])
                budget_list += metrics['budget_list']
                rmse_list += metrics['rmse_list']
                jsd_list += metrics['jsd_list']
                unc_list += metrics['unc_list']

            if curr_test > arg_eval.num_eval:
                perf_data = []
                for n in metric_names:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                idx = np.array(eval_num_list).argsort()
                avgjsd_list = np.array(avgjsd_list)[idx]
                avgunc_list = np.array(avgunc_list)[idx]
                stdjsd_list = np.array(stdjsd_list)[idx]
                stdunc_list = np.array(stdunc_list)[idx]
                minvisit_list = np.array(minvisit_list)[idx]
                avgvisit_list = np.array(avgvisit_list)[idx]
                stdvisit_list = np.array(stdvisit_list)[idx]
                budget_list = np.array(budget_list)
                rmse_list = np.array(rmse_list)
                jsd_list = np.array(jsd_list)
                unc_list = np.array(unc_list)
                break

        print(f'Graph {arg_eval.graph_size}, History {arg_eval.history_size}, #T {arg_eval.target_size},'
              f' Budget {arg_eval.budget_size}, K {arg_eval.k_size}, results:')
        for i in range(len(metric_names)):
            print(metric_names[i], ':\t', perf_data[i])
        if arg_eval.save_results:
            os.makedirs(arg_eval.result_path + '/metric', exist_ok=True)
            os.makedirs(arg_eval.result_path + '/traj', exist_ok=True)
            csv_filename_metric = f'{arg_eval.result_path}/metric/tgt{arg_eval.target_size}_speed{round(1/arg_eval.target_speed)}' \
                               f'_b{arg_eval.budget_size}_g{arg_eval.graph_size}_h{arg_eval.history_size}_metric.csv'
            csv_filename_traj = f'{arg_eval.result_path}/traj/tgt{arg_eval.target_size}_speed{round(1/arg_eval.target_speed)}' \
                               f'_b{arg_eval.budget_size}_g{arg_eval.graph_size}_h{arg_eval.history_size}_traj.csv'

            new_file = False if os.path.exists(csv_filename_traj) else True
            field_names = ['avgjsd', 'avgunc', 'stdjsd', 'stdunc', 'minnvisit', 'avgvist', 'stdvisit']
            with open(csv_filename_metric, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                metric_data = np.concatenate((avgjsd_list.reshape(-1, 1), avgunc_list.reshape(-1, 1), stdjsd_list.reshape(-1, 1),
                                              stdunc_list.reshape(-1, 1), minvisit_list.reshape(-1, 1), avgvisit_list.reshape(-1, 1),
                                              stdvisit_list.reshape(-1, 1)), axis=-1)
                writer.writerows(metric_data)

            new_file = False if os.path.exists(csv_filename_traj) else True
            field_names = ['budget', 'unc', 'jsd', 'rmse']
            with open(csv_filename_traj, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                traj_data = np.concatenate((budget_list.reshape(-1, 1), unc_list.reshape(-1, 1), jsd_list.reshape(-1, 1),
                                            rmse_list.reshape(-1, 1)), axis=-1)
                writer.writerows(traj_data)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_runners:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=len(arg_eval.cuda_devices) / arg_eval.num_meta)
class Runner:
    def __init__(self, meta_id):
        self.meta_id = meta_id
        self.device = torch.device('cuda') if arg_eval.use_gpu_runner else torch.device('cpu')
        self.local_net = AttentionNet(arg_eval.embedding_dim)
        self.local_net.to(self.device)

    def set_weights(self, weights):
        self.local_net.load_state_dict(weights)

    def job(self, global_weights, eval_number, config=None):
        print(f'\033[92mmeta{self.meta_id:02}:\033[0m eval {eval_number} starts')
        self.set_weights(global_weights)
        save_img = True if arg_eval.save_img_gap != 0 and eval_number % arg_eval.save_img_gap == 0 else False
        worker = WorkerEval(self.meta_id, self.local_net, eval_number, self.device, greedy=True, save_image=save_img, config=config)
        metrics = worker.run_episode(eval_number)
        return metrics, self.meta_id


if __name__ == '__main__':
    ray.init()
    print(f'#Evals: {arg_eval.num_eval}, #meta: {arg_eval.num_meta}')
    main()
    # Loop testing
    # for test_graph in [400]:
    #     for test_history in [200]:
    #         for test_tgt_number in [4]:
    #             for test_tgt_speed in [1/10]:
    #                 main(config=(test_graph, test_history, test_tgt_number, test_tgt_speed))
    #                 print('==================================')
