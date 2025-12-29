import time
import numpy as np
import pandas as pd

from bo.optimizer import Optimizer
from bo.optimizer_mf import OptimizerMF

def run_cas(args, f, prior=False, mf=False):
    res = pd.DataFrame(np.nan, index=np.arange(int(args.max_iters * args.batch_size)),
                       columns=['Index', 'LastValue', 'BestValue', 'Time'])  # 初始化一个空的dataframe来存运行记录
    if args.infer_noise_var:
        noise_variance = None
    else:
        noise_variance = f.lamda if hasattr(f, 'lamda') else None

    kwargs = {
        'length_max_discrete': args.ambulance,
        'length_init_discrete': args.ambulance,
        'p': f.p,  # number of ambulances
        'multiplier': 1.5,
        'failtol': 10,
    }

    if mf:
        optim = OptimizerMF(f.config, n_init=args.n_init, use_ard=args.ard, acq=args.acq,
                          kernel_type='transformed_overlap',
                          t_mat=f.t_mat, frac_j=f.frac_j_2,
                          noise_variance=noise_variance, **kwargs)
    else:
        optim = Optimizer(f.config, n_init=args.n_init, use_ard=args.ard, acq=args.acq,
                          kernel_type='transformed_overlap',
                          prior=prior, t_mat=f.t_mat, frac_j=f.frac_j_2,
                          noise_variance=noise_variance, **kwargs)

    time_start = time.time()
    for i in range(args.max_iters):
        start = time.time()
        x_next = optim.suggest(args.batch_size)
        if mf:
            y_next = f.compute(x_next[:, :-1], normalize=f.normalize)  # 这个就直接用我们的Larson Approx就行了，没有normalize
        else:
            y_next = f.compute(x_next, normalize=f.normalize)

        optim.observe(x_next, y_next)  # 把推荐出来的这个返回给optimizer (应该指的是cosmopolitan)

        end = time.time()
        if f.normalize:  # 如果normalize过了，在这里把它转回来。做了类似normal的假设。我们没有normalize
            Y = np.array(optim.casmopolitan.fX) * f.std + f.mean
        else:
            Y = np.array(optim.casmopolitan.fX)
        if Y[:i].shape[0]:  # 记录当前的情况
            # sequential
            if args.batch_size == 1:
                res.iloc[i, :] = [i, float(Y[-1]), float(np.min(Y[:i])), end - start]
            # batch
            else:
                for idx, j in enumerate(range(i * args.batch_size, (i + 1) * args.batch_size)):
                    res.iloc[j, :] = [j, float(Y[-idx]), float(np.min(Y[:i * args.batch_size])), end - start]
            argmin = np.argmin(Y[:i * args.batch_size])  # 最好的y的index

            print('Iter %d, Last X %s; fX:  %.4f. X_best: %s, fX_best: %.4f' % (i, x_next.flatten(), float(Y[-1]),
                                                                                ','.join([str(int(i)) for i in
                                                                                          optim.casmopolitan.X[
                                                                                          :i * args.batch_size][
                                                                                              argmin].flatten()]),
                                                                                Y[:i * args.batch_size][argmin]))
    time_end = time.time()
    print('time cost: ', time_end - time_start, 's\n')
    return optim.casmopolitan.fX, optim.casmopolitan.X
