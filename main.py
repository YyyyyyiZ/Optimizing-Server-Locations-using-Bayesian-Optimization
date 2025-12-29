from test_funcs import *
from CAS_runner import run_cas

import logging
import argparse
import pickle

import warnings
warnings.filterwarnings("ignore")

# Set up the objective function
parser = argparse.ArgumentParser('Run Experiments')
parser.add_argument('-p', '--problem', type=str, default='MRT_binary')
parser.add_argument('--max_iters', type=int, default=80, help='Maximum number of BO iterations.')
parser.add_argument('--lamda', type=float, default=1e-6, help='the noise to inject for some problems')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for BO.')
parser.add_argument('--n_trials', type=int, default=1, help='number of trials for the experiment')
parser.add_argument('--n_init', type=int, default=20, help='number of initialising random points')
parser.add_argument('--save_path', type=str, default='0512', help='save directory of the log files')
parser.add_argument('--ard', action='store_true', help='whether to enable automatic relevance determination')
parser.add_argument('-a', '--acq', type=str, default='ei', help='choice of the acquisition function.')
parser.add_argument('--random_seed_objective', type=int, default=20, help='The default value of 20 is provided also in COMBO')
parser.add_argument('-d', '--debug', action='store_true', help='Whether to turn on debugging mode (a lot of output will'
                                                               'be generated).')
parser.add_argument('--no_save', action='store_true', help='If activated, do not save the current run into a log folder.')
parser.add_argument('-k', '--kernel_type', type=str, default=None, help='specifies the kernel type')
parser.add_argument('--infer_noise_var', action='store_true')

parser.add_argument('-loc', type=int, default=17, help='# locations')
parser.add_argument('-mu', type=int, default=5, help='Mu')
parser.add_argument('-ambulance', type=int, default=9, help='# ambulances')    # fix_num

parser.add_argument('-opt', type=int, default=0, help='compute optimum using enumeration')
parser.add_argument('-p_median', type=int, default=0, help='p-median')
parser.add_argument('-GA', type=int, default=0, help='Genetic Algorithm')
parser.add_argument('-BOCS', type=int, default=0, help='BOCS')
parser.add_argument('-BOCS_sub', type=int, default=0, help='BOCS Submodular AFO')
parser.add_argument('-CAS', type=int, default=1, help='CASMOPOLITAN')

parser.add_argument('-GP_prior', type=int, default=0, help='GP with p-median prior')
parser.add_argument('-BOCS_prior', type=int, default=0, help='BOCS with p-median prior')
parser.add_argument('-MFBO', type=int, default=0, help='Multi-fidelity BO')

parser.add_argument('-st_paul', type=int, default=1, help='St. paul data')
parser.add_argument('--seed', type=int, default=0, help='random seed')


args = parser.parse_args()
options = vars(args)
print(options)

if args.debug:
    logging.basicConfig(level=logging.INFO)

# Sanity checks
assert args.acq in ['ucb', 'ei', 'thompson'], 'Unknown acquisition function choice ' + str(args.acq)

for t in range(0, args.n_trials, 1):
    # if t<3:
    #     continue
    print('----- Starting trial %d / %d -----' % ((t + 1), args.n_trials))

    kwargs = {}
    if args.random_seed_objective is not None:
        assert 1 <= int(args.random_seed_objective) <= 25
        args.random_seed_objective -= 1

    if t>0: args.opt = 0
    f = MRT(max_iter=args.max_iters, n_init=args.n_init, new_N = args.loc, p = args.ambulance, mu = args.mu,
            pmedian=args.p_median, ga = args.GA, bocs = args.BOCS, bocs_sub = args.BOCS_sub, opt = args.opt,
            bocs_prior=args.BOCS_prior, st_paul = args.st_paul, random_seed = t)

    cas_loc, cas = -1, -1
    if args.CAS:
        print('-' * 40 + "Running CASMOPOLITAN solution" + '-' * 40)
        cas, cas_loc = run_cas(args, f, prior=False)

    gp_prior_loc, gp_prior = -1, -1
    if args.GP_prior:
        print('-' * 40 + "Running CASMOPOLITAN with Prior solution" + '-' * 40)
        gp_prior, gp_prior_loc = run_cas(args, f, prior=True)

    mf_loc, mf = -1, -1
    if args.MFBO:
        print('-' * 40 + "Running Multi-fidelity BO solution" + '-' * 40)
        mf, mf_loc = run_cas(args, f, mf=True)


    # file_name = "res_new/{}/results_{}_{}_{}_priors.pkl".format(args.save_path,f.new_N, f.p, t)
    # with open(file_name, 'wb') as file:
    #         pickle.dump([cas_loc ,cas,
    #                      f.p_median_loc, f.p_median,
    #                      f.GA_loc, f.GA,
    #                      f.BOCS_loc, f.BOCS,
    #                      f.BOCS_sub_loc, f.BOCS_sub,
    #                      f.opt_loc, f.opt_f,
    #                      gp_prior_loc, gp_prior,
    #                      f.BOCS_prior_loc, f.BOCS_prior,
    #                      mf_loc, mf], file)
