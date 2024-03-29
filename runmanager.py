# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

from IPython.display import display, clear_output
import pandas as pd

from collections import OrderedDict
from collections import namedtuple
from itertools import product

import time
import json

__all__ =  ['RunBuilder', 
            'RunManager']

# Run builder - creates a run from a dictionary of hyperparameters
class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run',params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs

# Run manager - keeps track of the runs
class RunManager():

    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.epoch_test_loss = 0
        self.epoch_decision_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        # self.tb = None 
        
    # Begin run
    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        # self.tb = SummaryWriter(comment=f'-{run}')

    # End run
    def end_run(self):  
        # self.tb.close()
        self.epoch_count = 0

    # Begin epoch
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.epoch_test_loss = 0
        self.epoch_decision_loss = 0
        self.epoch_num_correct = 0

    # End epoch
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss
        valid_loss = self.epoch_val_loss
        test_loss = self.epoch_test_loss
        decision_loss = self.epoch_decision_loss

        #self.tb.add_scalar('Loss', loss, self.epoch_count)
        #self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        # for name, param in self.network.named_parameters():
            # self.tb.add_histogram(name, param, self.epoch_count)
            # self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["valid_loss"] = valid_loss
        results["test_loss"] = test_loss
        results["decision_loss"] = decision_loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df)

    
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)