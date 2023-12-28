from models import MLP
import torch, os

def create_and_load_model(PATH : str, in_channels : int):

    likelihood_fn = PATH.split('likelihood-fn-')[1].split('-dropout')[0]
    hidden_channels = [int(x) for x in PATH.split('/')[-2].split('mlp-')[1].split('-')]
    dropout_rate = int(PATH.split('dropout-rate-')[1].split('-')[0])
    linear_model = False  

    # print(likelihood_fn)
    # print(hidden_channels)
    # print(dropout_rate)
    # print(linear_model) 

    network = MLP(in_channels=in_channels,
                  hidden_channels= hidden_channels,
                  likelihood_fn=likelihood_fn,
                  dropout_rate= dropout_rate,
                  linear_model=linear_model)

    network.load_state_dict(torch.load(os.path.join(PATH,'model_best.pth.tar')))

    print('Model sucessfully loaded')

if __name__ == '__main__':
    create_and_load_model(PATH = '/Volumes/GoogleDrive/My Drive/1. Projects (short term efforts)/PhD/upper-indus-prec-bc/_experiments/2022-02-26_19-23-53_run-lr-0-005-batch-size-128-likelihood-fn-bgmm-hidden-channels-30-dropout-rate-0-linear-model-false-k-0',
                          in_channels=24)
        