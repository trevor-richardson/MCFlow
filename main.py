"""
Official implementation of MCFlow -
"""
import numpy as np
import torch
import argparse
import sys
import os
from models import InterpRealNVP
import util
from loader import DataLoader
from models import LatentToLatentApprox


def main(args):
    #initialize dataset class
    ldr = DataLoader(mode=0, seed=args.seed, path=args.dataset, drp_percent=args.drp_impt)
    data_loader = torch.utils.data.DataLoader(ldr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    num_neurons = int(ldr.train[0].shape[0])

    #Initialize normalizing flow model neural network and its optimizer
    flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args)
    nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.lr)

    #Initialize latent space neural network and its optimizer
    num_hidden_neurons = [int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]),  int(ldr.train[0].shape[0])]
    nn_model = LatentToLatentApprox(int(ldr.train[0].shape[0]), num_hidden_neurons).float()
    if args.use_cuda:
        nn_model.cuda()
    nn_optimizer = torch.optim.Adam([p for p in nn_model.parameters() if p.requires_grad==True], lr=args.lr)

    reset_scheduler = 2

    if args.dataset == 'news':
        print("\n****************************************")
        print("Starting OnlineNewsPopularity experiment\n")
    elif args.dataset == 'mnist':
        print("\n*********************************")
        print("Starting MNIST dropout experiment\n")
    else:
        print("Invalid dataset error")
        sys.exit()

    #Train and test MCFlow
    for epoch in range(args.n_epochs):
        util.endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, data_loader, args) #Train the MCFlow model

        with torch.no_grad():
            ldr.mode=1 #Use testing data
            te_mse, _ = util.endtoend_test(flow, nn_model, data_loader, args) #Test MCFlow model
            ldr.mode=0 #Use training data
            print("Epoch", epoch, " Test RMSE", te_mse**.5)

        if (epoch+1) % reset_scheduler == 0:
            #Reset unknown values in the dataset using predicted estimates
            if args.dataset == 'mnist':
                ldr.reset_img_imputed_values(nn_model, flow, args.seed, args)
            else:
                ldr.reset_imputed_values(nn_model, flow, args.seed, args)
            flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args) #Initialize brand new flow model to train on new dataset
            nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.lr)
            reset_scheduler = reset_scheduler*2

''' Run MCFlow experiment '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Reproducibility')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-nf-layers', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=500)
    parser.add_argument('--drp-impt', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-cuda', type=util.str2bool, default=True)
    parser.add_argument('--dataset', default='news', help='Two options: (1) letter-recogntion or (2) mnist')
    args = parser.parse_args()

    ''' Reproducibility '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ''' Cuda enabled experimentation check '''
    if not torch.cuda.is_available() or args.use_cuda==False:
        print("CUDA Unavailable. Using cpu. Check torch.cuda.is_available()")
        args.use_cuda = False

    if not os.path.exists('masks'):
        os.makedirs('masks')

    if not os.path.exists('data'):
        os.makedirs('data')

    main(args)
    print("Experiment completed")
