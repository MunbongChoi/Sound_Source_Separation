
from conv_tasnet import conv_tasnet, train, utils, 
import biocppnet

def load_model(model_name, input_dim):
    if model_name == 'conv_tasnet':
        train
        N = 256	
        L = 20	
        B = 256	
        H = 512	
        P = 3	
        X = 8	
        R = 4
        
        model = conv_tasnet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       args.C, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)
        
        model = ConvTasNet(N, L, B, H, P, X, R,
                       C, norm_type=norm_type, causal=causal,
                       mask_nonlinear=mask_nonlinear)
        return conv_tasnet
    elif model_name == 'biocppnet':
        return Model2(input_dim)