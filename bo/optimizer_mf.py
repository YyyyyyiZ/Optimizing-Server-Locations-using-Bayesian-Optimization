from .localbo_cat_mf import CASMOPOLITANCatMF
from .localbo_utils import binary_init, sample_neighbour_ordinal_binary_constraint, train_gp

import numpy as np
import scipy.stats as ss
from copy import deepcopy
import torch


class OptimizerMF:
    def __init__(self, config, n_init=None, wrap_discrete=True, guided_restart=True,
                 t_mat=None, frac_j=None, cost_ratio=5.0, **kwargs):

        self.config = config.astype(int)
        self.true_dim = len(config)
        self.kwargs = kwargs
        self.n_onehot = int(np.sum(config))
        self.lb = np.zeros(self.n_onehot)
        self.ub = np.ones(self.n_onehot)
        self.dim = len(self.lb)
        self.max_evals = np.iinfo(np.int32).max
        self.batch_size = None
        self.history = []
        self.wrap_discrete = wrap_discrete
        self.cat_dims = self.get_dim_info(config)
        self.cost_ratio = cost_ratio

        self.t_mat = t_mat
        self.frac_j = frac_j

        # 多保真度特有属性
        self.fidelity_dim = 1  # 保真度维度 (0=低, 1=高)
        self.current_fidelity = 0.3  # 初始低保真度概率

        # 初始化优化器
        self.casmopolitan = CASMOPOLITANCatMF(
            dim=self.true_dim + self.fidelity_dim,  # 增加保真度维度
            n_init=n_init if n_init is not None else 2 * self.true_dim + 1,
            max_evals=self.max_evals,
            batch_size=1,
            verbose=False,
            t_mat=self.t_mat, frac_j=self.frac_j,
            config=np.append(self.config, 2),  # 保真度作为二元变量
            **kwargs
        )

        # 引导重启相关
        self.guided_restart = guided_restart
        self.best_X_each_restart, self.best_fX_each_restart = None, None
        self.auxiliary_gp = None

    def restart(self):
        """多保真度版本的重启方法"""
        if self.guided_restart and len(self.casmopolitan._fX):
            # 收集各重启周期的最佳点
            best_idx = self.casmopolitan._fX.argmin()
            if self.best_fX_each_restart is None:
                self.best_fX_each_restart = deepcopy(self.casmopolitan._fX[best_idx])
                self.best_X_each_restart = deepcopy(self.casmopolitan._X[best_idx])
            else:
                self.best_fX_each_restart = np.vstack(
                    (self.best_fX_each_restart, deepcopy(self.casmopolitan._fX[best_idx])))
                self.best_X_each_restart = np.vstack(
                    (self.best_X_each_restart, deepcopy(self.casmopolitan._X[best_idx])))

            # 训练辅助GP (多保真度版本)
            X_tr_torch = torch.tensor(self.best_X_each_restart, dtype=torch.float32).reshape(-1,
                                                                                             self.true_dim + self.fidelity_dim)
            fX_tr_torch = torch.tensor(self.best_fX_each_restart, dtype=torch.float32).view(-1)

            # 使用多保真度GP训练
            self.auxiliary_gp = train_gp(X_tr_torch, fX_tr_torch, False, 300, mf=True, t_mat=self.t_mat, frac_j=self.frac_j)


            # 生成初始点 (包含保真度)
            X_init = np.zeros((self.casmopolitan.n_init, self.true_dim + self.fidelity_dim))
            X_init[:, :-1] = binary_init(self.casmopolitan.n_init, self.true_dim, self.kwargs['p'])
            X_init[:, -1] = np.random.binomial(1, self.current_fidelity, size=self.casmopolitan.n_init)

            # 使用辅助GP选择最佳初始点
            with torch.no_grad():
                self.auxiliary_gp.eval()
                X_init_torch = torch.tensor(X_init, dtype=torch.float32)
                y_cand_mean = self.auxiliary_gp(X_init_torch).mean.cpu().detach().numpy()
                y_cand_var = self.auxiliary_gp(X_init_torch).variance.cpu().detach().numpy()
                y_cand = y_cand_mean - 1.96 * np.sqrt(y_cand_var)  # LCB

            indbest = np.argmin(y_cand)
            centre = deepcopy(X_init[indbest, :])
            self.X_init = np.zeros((self.casmopolitan.n_init, self.true_dim + self.fidelity_dim))
            self.X_init[0, :] = deepcopy(centre)

            # 生成邻域点
            for i in range(1, self.casmopolitan.n_init):
                design_vars = sample_neighbour_ordinal_binary_constraint(centre[:-1], self.config, 4)
                fidelity = np.random.binomial(1, self.current_fidelity)
                self.X_init[i, :] = np.append(design_vars, fidelity)

        else:
            # 常规重启
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))

            # 生成初始点 (包含保真度)
            self.X_init = np.zeros((self.casmopolitan.n_init, self.true_dim + self.fidelity_dim))
            self.X_init[:, :-1] = binary_init(self.casmopolitan.n_init, self.true_dim, self.kwargs['p'])
            self.X_init[:, -1] = np.random.binomial(1, self.current_fidelity, size=self.casmopolitan.n_init)

    def suggest(self, n_suggestions=1):
        """生成建议点 (包含保真度选择)"""
        if self.batch_size is None:
            self.batch_size = n_suggestions
            self.casmopolitan.batch_size = n_suggestions
            self.casmopolitan.n_init = max([self.casmopolitan.n_init, self.batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.true_dim + self.fidelity_dim))

        # 从初始点中选择
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]

        # 从优化器中获取剩余点
        n_adapt = n_suggestions - n_init
        if n_adapt > 0 and len(self.casmopolitan._X) > 0:
            X = deepcopy(self.casmopolitan._X)
            fX = deepcopy(self.casmopolitan._fX).ravel()
            X_next[-n_adapt:, :] = self.casmopolitan._create_and_select_candidates(
                X, fX,
                length=self.casmopolitan.length_discrete,
                n_training_steps=300,
                hypers={}
            )[-n_adapt:, :]

        # 动态调整保真度概率 (基于历史表现)
        if len(self.casmopolitan._fX) > 10:
            hf_mask = self.casmopolitan._X[:, -1] > 0.5
            if np.sum(hf_mask) > 0:
                hf_improvement = np.mean(self.casmopolitan._fX[hf_mask]) - np.mean(self.casmopolitan._fX[~hf_mask])
                self.current_fidelity = np.clip(0.1 + 0.8 * (1 - np.exp(-hf_improvement)), 0.1, 0.9)

        return X_next

    def observe(self, X, y):
        """观察评估结果"""
        assert len(X) == len(y)
        XX = X
        yy = np.array(y)[:, None]

        if len(self.casmopolitan._fX) >= self.casmopolitan.n_init:
            self.casmopolitan._adjust_length(yy)

        self.casmopolitan.n_evals += self.batch_size
        self.casmopolitan._X = np.vstack((self.casmopolitan._X, deepcopy(XX)))
        self.casmopolitan._fX = np.vstack((self.casmopolitan._fX, deepcopy(yy.reshape(-1, 1))))
        self.casmopolitan.X = np.vstack((self.casmopolitan.X, deepcopy(XX)))
        self.casmopolitan.fX = np.vstack((self.casmopolitan.fX, deepcopy(yy.reshape(-1, 1))))

        # 检查是否需要重启
        if (self.casmopolitan.length <= self.casmopolitan.length_min or
                self.casmopolitan.length_discrete <= self.casmopolitan.length_min_discrete):
            self.restart()

    def get_dim_info(self, n_categories):
        """获取维度信息 (与原始版本相同)"""
        dim_info = []
        offset = 0
        for i, cat in enumerate(n_categories):
            dim_info.append(list(range(offset, offset + cat)))
            offset += cat
        return dim_info
