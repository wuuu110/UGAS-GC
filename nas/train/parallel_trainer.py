from concurrent.futures import ThreadPoolExecutor, as_completed
from torch_geometric.graphgym.config import cfg


class ParallelTrainer():
    def __init__(self, parallel_task: list):
        self.parallel_task = parallel_task
    
    def run(self):
        # Initialize ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cfg.nas.parallel_num) as executor:
            # Submit training tasks
            futures = [executor.submit(trainer.run) for trainer in self.parallel_task]
            
        return [future.result() for future in futures]