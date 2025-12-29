from copy import deepcopy

import numpy as np
import scipy.stats as ss
from .localbo_cat import CASMOPOLITANCat
from .localbo_utils import binary_init, sample_neighbour_ordinal_binary_constraint
import torch


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


class Optimizer:

    def __init__(self, config,
                 n_init: int = None,  # 初始化采样的数量，目前默认为20
                 wrap_discrete: bool = True,
                 guided_restart: bool = True,
                 prior: bool = False, t_mat=None, frac_j=None,
                 **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        config: list. e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
            being 2, 3, 4, and 5 respectively.
        guided_restart: whether to fit an auxiliary GP over the best points encountered in all previous restarts, and
            sample the points with maximum variance for the next restart.
        global_bo: whether to use the global version of the discrete GP without local modelling
        """

        # Maps the input order.
        self.config = config.astype(int) # 从问题f中来，是个list。目前我们是[N,N...,N]共p个，按理说我们应该是[2,2,..2]共N个，但是有个限制是共p个1。可能需要改。
        self.true_dim = len(config) # 问题的dimension
        self.kwargs = kwargs        # 其他读进来的参数
        # Number of one hot dimensions
        self.n_onehot = int(np.sum(config)) # 如果变成one-hot是多少个dimension
        # One-hot bounds
        self.lb = np.zeros(self.n_onehot)
        self.ub = np.ones(self.n_onehot)
        self.dim = len(self.lb)
        # True dim is simply th`e number of parameters (do not care about one-hot encoding etc).
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None  # 对于optimizer，初始化为none。因为还没进入cosmopolitan，那个里面batch_size不能是none。
        self.history = []
        self.wrap_discrete = wrap_discrete
        self.cat_dims = self.get_dim_info(config)
        self.prior = prior
        self.t_mat = t_mat
        self.frac_j = frac_j

        self.casmopolitan = CASMOPOLITANCat( # 这个应该是我们用到的算法。里面用到的主要是_create_and_select_candidates，会被调用
            dim=self.true_dim,              # 问题的dimension
            n_init=n_init if n_init is not None else 2 * self.true_dim + 1,  # 初始化采样的数量，目前默认为20。如果给了就按给的，没给就是2*dim + 1
            max_evals=self.max_evals,
            batch_size=1,  # We need to update this later。# 之后应该可以选多个，不知道是否会对提升效果有用。
            verbose=False,  # 这个应该是是否print吧
            config=self.config,
            prior = self.prior, t_mat = self.t_mat, frac_j = self.frac_j,
            **kwargs
        )

        # Our modification: define an auxiliary GP  # 感觉这里之前是他们和别人一样的，这个是他们加进来的
        self.guided_restart = guided_restart    # 这个是个boolean，是否需要guided restart
        # keep track of the best X and fX in each restart
        self.best_X_each_restart, self.best_fX_each_restart = None, None
        self.auxiliary_gp = None

    def restart(self): # restart是在suggest里面被调用。
        from .localbo_utils import train_gp

        if self.guided_restart and len(self.casmopolitan._fX):  # 只有TR shrink到1之后才会触发这个。_fx是_restart里面初始化的list，里面存的是当前restart下最好的f(x)。

            best_idx = self.casmopolitan._fX.argmin()           # 当前list里最好的f(x)的id
            # Obtain the best X and fX within each restart (bo._fX and bo._X get erased at each restart,
            # but bo.X and bo.fX always store the full history
            # 把每次restart后最好的X和f(X)存到best_fX_each_restart这个nparray里
            if self.best_fX_each_restart is None:  
                self.best_fX_each_restart = deepcopy(self.casmopolitan._fX[best_idx])
                self.best_X_each_restart = deepcopy(self.casmopolitan._X[best_idx])
            else:
                self.best_fX_each_restart = np.vstack((self.best_fX_each_restart, deepcopy(self.casmopolitan._fX[best_idx])))
                self.best_X_each_restart = np.vstack((self.best_X_each_restart, deepcopy(self.casmopolitan._X[best_idx])))

            X_tr_torch = torch.tensor(self.best_X_each_restart, dtype=torch.float32).reshape(-1, self.true_dim)  # x变成tensor
            fX_tr_torch = torch.tensor(self.best_fX_each_restart, dtype=torch.float32).view(-1)                  # f(x)变成tensor
            # Train the auxiliary # 这里训练的是只把最好的X和fX进行了GP训练。如果某一次restart得到的最好的和之前一样，就加进X和fX一个random的。
            # 用来选TR center
            self.auxiliary_gp = train_gp(X_tr_torch, fX_tr_torch, False, 300, prior=self.prior, t_mat=self.t_mat, frac_j=self.frac_j)  # train了个auxiliary global GP， 用每次restart里最好的X和f(X)。这里use_ard=False, num_steps=300。
            X_init = binary_init(self.casmopolitan.n_init, self.true_dim, self.kwargs['p'])
            print('X_init', X_init)
            with torch.no_grad():
                self.auxiliary_gp.eval()
                X_init_torch = torch.tensor(X_init, dtype=torch.float32)   # 转成tensor。这个是我们的要去试的X，其中最好的预测的f(X)的LCB会被作为新的center。
                # LCB-sampling
                y_cand_mean, y_cand_var = self.auxiliary_gp(               # 把需要测的posterior mean和variance从GP里得到
                    X_init_torch).mean.cpu().detach().numpy(), self.auxiliary_gp(
                    X_init_torch).variance.cpu().detach().numpy()
                y_cand = y_cand_mean - 1.96 * np.sqrt(y_cand_var)          # UCB的反向，LCB。这个好像就是文中(3)，会作为TR center

            self.X_init = np.ones((self.casmopolitan.n_init, self.true_dim))  # Class里重新初始化
            indbest = np.argmin(y_cand)                                     # 目前最好的y的index，用LCB选的
            # The initial trust region centre for the test restart
            centre = deepcopy(X_init[indbest, :])                           # 初始的TR center，目前好像是全1
            # The centre is the first point to be evaluated
            self.X_init[0, :] = deepcopy(centre)                            # 这个center作为新的TR center去测试

            # 这里是从TR region里采样n_init个去侧，最开始的那几个。给定center。然后把后面2到20用center中间的改掉
            for i in range(1, self.casmopolitan.n_init):
                # Randomly sample within the initial trust region length around the centre
                # self.X_init[i, :] = deepcopy(                               
                #     random_sample_within_discrete_tr_ordinal(centre, self.casmopolitan.length_init_discrete, self.config))
                self.X_init[i, :] = deepcopy(sample_neighbour_ordinal_binary_constraint(centre, self.config, 4))           # 默认hamming distance为4

            # 这里开始是cosmopolitan的restart
            self.casmopolitan._restart()                                    # cosmopolitan的restart。X，f(X),n_success, n_fail都归空或0
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))     # 这个重新再更新X个f(X)
            self.casmopolitan._fX = np.zeros((0, 1))
            del X_tr_torch, fX_tr_torch, X_init_torch                       # 把一些用完的删掉？

        else:  # 这里是如果不restart的情况，或者一开始啥都没有，就直接进入这个。
            # If guided restart is not enabled, simply sample a number of points equal to the number of evaluated
            # 直接cosmopolitan的restart
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            # X_init 的restart。开始选东西进来。
            ###### 我先把这些删掉了 #######
            # print('self.casmopolitan.n_init, self.dim', self.casmopolitan.n_init, self.dim)
            # X_init = latin_hypercube(self.casmopolitan.n_init, self.dim)
            # print(X_init)
            # self.X_init = from_unit_cube(X_init, self.lb, self.ub)
            # print(self.X_init)
            # if self.wrap_discrete:
            #     self.X_init = self.warp_discrete(self.X_init, )
            #     print('wrap', self.X_init)
            # self.X_init = onehot2ordinal(self.X_init, self.cat_dims)
            self.X_init = binary_init(self.casmopolitan.n_init, self.true_dim, self.kwargs['p'])


    def suggest(self, n_suggestions=1):  # 这个是最先用到的。
        if self.batch_size is None:  # Remember the batch size on the first call to suggest  # 对于optimizer初始化为none。所以会先进入这个if语句。
            self.batch_size = n_suggestions   # 目前是suggest一个。                            # 由于它，应该只有第一遍会进入这个循环
            self.casmopolitan.batch_size = n_suggestions  # 目前是suggest一个。更新casmopolitan里的batch_size
            # self.bo.failtol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.casmopolitan.n_init = max([self.casmopolitan.n_init, self.batch_size])  # 初始化采样的数量，目前默认为20。按照目前肯定比1大，所以就是20
            self.restart()  # 先来一步restart。这里面很多很多东西，包括主要X_init和之前存储过的一些东西的更新

        X_next = np.zeros((n_suggestions, self.true_dim))           # X_next的placeholder

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)               # self.X_init最开始是在self.restart设定的，通过center选的TR的点。
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])     # X_next从self.X_init得到，把前n_suggestions拿到
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        # 这些感觉是其他init的但是少于suggest的
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.casmopolitan._X) > 0:  # Use random points if we can't fit a GP
                X = deepcopy(self.casmopolitan._X)
                fX = copula_standardize(deepcopy(self.casmopolitan._fX).ravel())  # Use Copula  # 像是copula是一个随机拿点的办法
                X_next[-n_adapt:, :] = self.casmopolitan._create_and_select_candidates(X, fX,
                                                                                       length=self.casmopolitan.length_discrete, # 这个根据问题而来，我们问题定义为20
                                                                                       n_training_steps=300,
                                                                                       hypers={})[-n_adapt:, :]

        suggestions = X_next                   # X_next就是我们的suggestions
        return suggestions

    def observe(self, X, y): 
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like           # 是dict吗？我怎么看就是个nparray呢
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)
        # XX = torch.cat([ordinal2onehot(x, self.n_categories) for x in X]).reshape(len(X), -1)
        XX = X
        yy = np.array(y)[:, None]
        # if self.wrap_discrete:
        #     XX = self.warp_discrete(XX, )

        if len(self.casmopolitan._fX) >= self.casmopolitan.n_init:      # 如果当前cosmopolitan存的fX比init的数量多，就调整下TR长度
            self.casmopolitan._adjust_length(yy)                        # 看看是否调整TR的大小。成功次数多了就可以调整。里面都包括了

        self.casmopolitan.n_evals += self.batch_size                    # 把evaluation数量增加，我们应该是+1
        self.casmopolitan._X = np.vstack((self.casmopolitan._X, deepcopy(XX)))  # 把这个新X放进来
        self.casmopolitan._fX = np.vstack((self.casmopolitan._fX, deepcopy(yy.reshape(-1, 1))))  # 把新的y放进来
        self.casmopolitan.X = np.vstack((self.casmopolitan.X, deepcopy(XX)))    # 对于整个模型也放进来
        self.casmopolitan.fX = np.vstack((self.casmopolitan.fX, deepcopy(yy.reshape(-1, 1))))

        # Check for a restart  # 如果TR length太小了就restart
        if self.casmopolitan.length <= self.casmopolitan.length_min or self.casmopolitan.length_discrete <= self.casmopolitan.length_min_discrete:
            self.restart()

    def warp_discrete(self, X, ):  # 有一些给了0，其余的给了1。但具体怎么弄的还要看看。

        X_ = np.copy(X)
        # Process the integer dimensions
        if self.cat_dims is not None:                       # 如果有categorical
            for categorical_groups in self.cat_dims:
                max_col = np.argmax(X[:, categorical_groups], axis=1)
                X_[:, categorical_groups] = 0
                for idx, g in enumerate(max_col):
                    X_[idx, categorical_groups[g]] = 1
        return X_

    def get_dim_info(self, n_categories):
        dim_info = []
        offset = 0
        for i, cat in enumerate(n_categories):
            dim_info.append(list(range(offset, offset + cat)))
            offset += cat
        return dim_info
