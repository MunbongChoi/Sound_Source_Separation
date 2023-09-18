import torch
from Model.conv_tasnet.solver import Solver
from Model.conv_tasnet.conv_tasnet import ConvTasNet

def load_model(model_name, input_dim):
    if model_name == 'conv_tasnet':
        N = 256	
        L = 20	
        B = 256	
        H = 512	
        P = 3	
        X = 8	
        R = 4
        C = 2
        use_cuda = True
        optimizer = 'sgd'
        norm_type = 'gLN'
        lr = 0.0001
        momentum = 0.0
        l2 = 0.0
        model = ConvTasNet(N, L, B, H, P, X, R,
                        C, norm_type=norm_type, causal=False,
                        mask_nonlinear='relu')
        print(model)
        if use_cuda:
            model = torch.nn.DataParallel(model)
            model.cuda()
        # optimizer
        if optimizer == 'sgd':
            optimizier = torch.optim.SGD(model.parameters(),
                                        lr=lr,
                                        momentum=momentum,
                                        weight_decay=l2)
        elif optimizer == 'adam':
            optimizier = torch.optim.Adam(model.parameters(),
                                        lr=lr,
                                        weight_decay=l2)
        # solver
        solver = Solver(data, model, optimizier, args)
        solver.train()
    
    elif model_name == 'biocppnet':
        return Model2(input_dim)