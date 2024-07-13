import argparse
import distutils
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filein', type=str, help='Input file')
    parser.add_argument('--p', dest='p', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('--c', dest='c', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('--cosmo', type=str, help='Cosmology type')
    parser.add_argument('--num_samples', type=int, default = 1000, help='Number of samples')
    parser.add_argument('--num_warmup', type=int, default = 1000, help='Number of warmup samples')
    parser.add_argument('--num_chains', type=int, default = 2, help='Number of chains')
    parser.add_argument('--N',type=int, default=10, help='Optional argument N')
    parser.add_argument('--target',type=float, default=0.8, help='Optional argument target_accept_prob')
    parser.add_argument('--cov_redshift', dest='cov_redshift', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    parser.add_argument('--batch', dest='batch', type=lambda x:bool(distutils.util.strtobool(x)),default=True)
    parser.add_argument('--wa_const', dest='wa_const', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    parser.add_argument('--w0_const', dest='w0_const', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    parser.add_argument('--key',type=int, default=0, help='Optional argument key_int')
    parser.add_argument('--GMM_zL', action='store_true', help='Optional argument GMM_zL')
    parser.add_argument('--GMM_zS', action='store_true', help='Optional argument GMM_zS')
    parser.add_argument('--fix_GMM', action='store_true', help='Optional argument fix_GMM')
    args = parser.parse_args()
    return args

argv =  argument_parser()
print(argv)