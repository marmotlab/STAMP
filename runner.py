import torch
import ray
from network import AttentionNet
from worker import Worker
from arguments import arg


@ray.remote(num_cpus=1, num_gpus=len(arg.cuda_devices) / arg.num_meta)
class Runner(object):
    def __init__(self, meta_id):
        self.meta_id = meta_id
        self.device = torch.device('cuda') if arg.use_gpu_runner else torch.device('cpu')
        self.local_net = AttentionNet(arg.embedding_dim)
        self.local_net.to(self.device)

    def get_weights(self):
        return self.local_net.state_dict()

    def set_weights(self, weights):
        self.local_net.load_state_dict(weights)

    def job(self, global_weights, episode_number, budget_size, graph_size, history_size, target_size):
        print(f'\033[92mmeta{self.meta_id:02}:\033[0m episode {episode_number} starts')
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)
        save_img = True if arg.save_img_gap != 0 and episode_number % arg.save_img_gap == 0 and episode_number != 0 else False
        worker = Worker(self.meta_id, self.local_net, episode_number, budget_size, graph_size, history_size, target_size,
                        self.device, greedy=False, save_image=save_img)
        job_results, metrics = worker.run_episode(episode_number)
        return job_results, metrics



if __name__=='__main__':
    pass
