from copy import deepcopy
import numpy as np
import torch
import gpytorch
from .localbo_utils import train_gp, random_sample_within_discrete_tr_ordinal
from .localbo_cat import CASMOPOLITANCat


class CASMOPOLITANCatMF(CASMOPOLITANCat):
    def __init__(self, dim, n_init, max_evals, config, t_mat, frac_j, batch_size=1, verbose=True,
                 use_ard=True, max_cholesky_size=2000, n_training_steps=50,
                 min_cuda=1024, device="cpu", dtype="float32", acq='thompson',
                 kernel_type='transformed_overlap', cost_ratio=5.0, **kwargs):
        """
        新增参数:
        cost_ratio: 高保真与低保真评估成本比
        """
        # 调用父类初始化 (注意dim已包含保真度维度)
        super().__init__(
            dim=dim, n_init=n_init, max_evals=max_evals, config=config,
            batch_size=batch_size, verbose=verbose, use_ard=use_ard,
            max_cholesky_size=max_cholesky_size, n_training_steps=n_training_steps,
            min_cuda=min_cuda, device=device, dtype=dtype, acq=acq,
            kernel_type=kernel_type,  **kwargs
        )

        # 多保真度特有属性
        self.t_mat = t_mat
        self.frac_j = frac_j
        self.cost_ratio = cost_ratio
        self.current_fidelity = 0.3  # 初始高保真评估概率

    def _create_and_select_candidates(self, X, fX, length, n_training_steps, hypers, return_acq=False):
        """修改候选点生成以处理保真度维度"""
        # 分离设计变量和保真度
        design_vars = X[:, :-1] if X.shape[1] == self.dim else X
        fidelity = X[:, -1:] if X.shape[1] == self.dim else np.zeros((len(X), 1))

        # 训练多保真度GP
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)

            # 使用多保真度训练函数
            gp = train_gp(                                              # 这里用来训练的是所有真实测过的x和fX
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type, mf=True,
                prior=self.prior, t_mat=self.t_mat, frac_j=self.frac_j,
                noise_variance=self.kwargs['noise_variance'] if
                'noise_variance' in self.kwargs else None
            )
            hypers = gp.state_dict()

        # 获取当前最佳点 (忽略保真度维度)
        x_center = design_vars[fX.argmin().item(), :][None, :]

        # 修改采集函数以考虑保真度成本
        def _ei(X, augmented=True):
            """多保真度EI采集函数"""
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if X.dim() == 1:
                X = X.reshape(1, -1)

            # 分离设计变量和保真度
            X_design = X[:, :-1]
            X_fidelity = X[:, -1:]

            # 获取预测
            preds = gp(X)
            mean, std = -preds.mean, preds.stddev

            # 计算基础EI
            mu_star = -gp.likelihood(gp(torch.tensor(
                np.hstack([x_center, np.ones((1, 1))]),  # 使用高保真度作为参考
                dtype=torch.float32
            ))).mean

            u = (mean - mu_star) / std
            ucdf = torch.distributions.Normal(0, 1).cdf(u)
            updf = torch.exp(torch.distributions.Normal(0, 1).log_prob(u))
            ei = std * updf + (mean - mu_star) * ucdf

            # 成本调整
            cost = 1 + (self.cost_ratio - 1) * X_fidelity
            ei = ei / cost

            if augmented:
                sigma_n = gp.likelihood.noise
            ei *= (1. - torch.sqrt(sigma_n) / torch.sqrt(sigma_n + std ** 2))

            return ei

        # 生成候选点 (包含保真度)
        if self.acq in ['ei', 'ucb']:
            if self.batch_size == 1:
                # 修改局部搜索以生成含保真度的候选点
                X_next, acq_next = self._local_search_with_fidelity(
                    x_center[0], _ei, length
                )
            else:
                # 批处理模式 (实现类似但需要处理多个点)
                raise NotImplementedError("Batch mode not implemented for MF")
        elif self.acq == 'thompson':
            # 修改Thompson采样以包含保真度
            X_next, acq_next = self._thompson_with_fidelity(gp, x_center[0], length)
        else:
            raise ValueError(f'Unknown acquisition function {self.acq}')

        # 动态调整保真度概率
        if len(fX) > 10:
            self._adjust_fidelity_probability(X, fX)

        if return_acq:
            return X_next, acq_next
        return X_next

    def _local_search_with_fidelity(self, x_center, acq_func, length):
        """生成含保真度的候选点"""
        from .localbo_utils import local_search

        # 首先生成设计变量候选
        design_candidates, _ = local_search(
            x_center,
            lambda x: acq_func(torch.cat([
                torch.tensor(x).float().reshape(1, -1),
                torch.ones(1, 1)  # 假设高保真评估
            ], dim=1)),
            self.config, length, 3, 1
        )

        # 为每个设计候选生成保真度
        candidates = []
        for design in design_candidates:
            # 随机决定保真度 (基于当前概率)
            fidelity = np.random.binomial(1, self.current_fidelity)
            candidates.append(np.append(design, fidelity))

        return np.array(candidates), None

    def _thompson_with_fidelity(self, gp, x_center, length):
        """多保真度Thompson采样"""
        n_cand = min(100 * self.dim, 5000)

        # 生成候选点 (包含保真度)
        X_cand = np.zeros((n_cand, self.dim))
        X_cand[:, :-1] = np.array([
            random_sample_within_discrete_tr_ordinal(x_center, length, self.config[:-1])
            for _ in range(n_cand)
        ])
        X_cand[:, -1] = np.random.binomial(1, self.current_fidelity, size=n_cand)

        # 评估并选择最佳候选
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand, dtype=torch.float32)

            # y_cand = train_gp(
            #     train_x=X_cand_torch,
            #     train_y=torch.zeros(n_cand, 1),  # 虚拟目标值
            #     use_ard=self.use_ard,
            #     num_steps=self.n_training_steps,
            #     hypers={},
            #     kern=self.kernel_type
            # ).likelihood(train_gp(
            #         train_x=X_cand_torch,
            #         train_y=torch.zeros(n_cand, 1),  # 虚拟目标值
            #         use_ard=self.use_ard,
            #         num_steps=self.n_training_steps,
            #         hypers={},
            #         kern=self.kernel_type
            #     )(X_cand_torch)
            # ).sample(torch.Size([self.batch_size])).t().cpu().numpy()

            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        best_indices = np.argpartition(y_cand.flatten(), self.batch_size)[:self.batch_size]
        return X_cand[best_indices], y_cand[best_indices]

    def _adjust_fidelity_probability(self, X, fX):
        """根据历史表现调整保真度概率"""
        hf_mask = X[:, -1] > 0.5
        if np.sum(hf_mask) > 0 and np.sum(~hf_mask) > 0:
            # 计算高保真评估的平均改进
            hf_improvement = np.median(fX[~hf_mask]) - np.median(fX[hf_mask])
            # 动态调整概率 (逻辑函数转换)
            self.current_fidelity = 1 / (1 + np.exp(-hf_improvement * 2))
            self.current_fidelity = np.clip(self.current_fidelity, 0.1, 0.9)