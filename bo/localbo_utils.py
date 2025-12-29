import math
import logging

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from collections import Callable
import random
from copy import deepcopy
import time
from bo.kernels import *
# debug
from test_funcs import *

def binary_init(n_init, dim, p): 
    # p is number of units
    np.random.seed(random.randint(0, 1e6))
    rand_indices = [np.random.choice(range(dim), size=p, replace=False) for _ in range(n_init)]
    X_init = np.zeros((n_init, dim))
    for i,r in enumerate(rand_indices):
        X_init[i,r] = 1
    # X_init = torch.tensor(X_init) # 应该不需要它是torch
    return X_init

def onehot2ordinal(x, categorical_dims):
    """Convert one-hot representation of strings back to ordinal representation."""
    from itertools import chain
    if x.ndim == 1:
        x = x.reshape(1, -1)
    categorical_dims_flattned = list(chain(*categorical_dims))
    # Select those categorical dimensions only
    x = x[:, categorical_dims_flattned]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    res = torch.zeros(x.shape[0], len(categorical_dims), dtype=torch.float32)
    for i, var_group in enumerate(categorical_dims):
        res[:, i] = torch.argmax(x[:, var_group], dim=-1).float()
    return res


def ordinal2onehot(x, n_categories):
    """Convert ordinal to one-hot"""
    res = np.zeros(np.sum(n_categories))
    offset = 0
    for i, cat in enumerate(n_categories):
        res[offset + int(x[i])] = 1
        offset += cat
    return torch.tensor(res)


# how to update posterior
class PMedianMean(gpytorch.means.Mean):
    """
    Custom mean function for the p-median problem.
    Computes the weighted response time objective as the mean function.

    Args:
        t_mat (torch.Tensor): K x N matrix of response times (t_ij)
        frac_j (torch.Tensor): K-length tensor of weights for each demand point
        input_size (int): Dimension of the input (number of potential facilities)
    """

    def __init__(self, t_mat, frac_j):
        super().__init__()
        # Convert numpy arrays to torch tensors if needed
        if isinstance(t_mat, np.ndarray):
            t_mat = torch.tensor(t_mat, dtype=torch.float32)
        if isinstance(frac_j, np.ndarray):
            frac_j = torch.tensor(frac_j, dtype=torch.float32)

        # Register buffers for the constant parameters
        self.register_buffer('t_mat', t_mat)
        self.register_buffer('frac_j', frac_j)
        self.t_mat = t_mat
        self.frac_j = frac_j

        # We'll use a linear transformation of the input to ensure valid probabilities
        self.transform = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Compute the mean function for the p-median problem.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size x input_size)
                             Represents potential solutions (binary or continuous)

        Returns:
            torch.Tensor: Objective values for each input solution
        """
        # Ensure input is 2D (batch_size x input_size)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        objectives = torch.zeros(batch_size, device=x.device)
        # Apply sigmoid to get values between 0 and 1 (for continuous relaxation)
        x_probs = self.transform(x)

        for k in range(batch_size):
            # Threshold to get binary decisions (can use different thresholds)
            open_facilities = x_probs[k] > 0.5
            # Get response times for open facilities only
            open_t_mat = self.t_mat[:, open_facilities]
            # Find nearest facility for each demand point
            min_values, nearest_indices = torch.min(open_t_mat, dim=1)
            # Compute weighted sum
            objectives[k] = torch.sum(self.frac_j * min_values)
        return objectives

class MultiFidelityGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kern, likelihood,
                 low_fidelity_model, outputscale_constraint, ard_dims, cat_dims=None, use_ard=False):
        super().__init__(train_x, train_y, likelihood)

        self.low_fidelity_model = low_fidelity_model
        self.high_fidelity_gp = None

        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        self.kern = kern

        self.train_x = train_x
        self.train_y = train_y

        # 保真度参数转换 (假设最后一个维度是保真度)
        self.fidelity_transform = torch.nn.Sigmoid()
        self.covar_module = self._create_kernel(use_ard, train_x.size(-1) - 1, outputscale_constraint)
        self.mean_module = ConstantMean()

    def _create_kernel(self, use_ard, input_dim, outputscale_constraint):
        """创建协方差内核"""
        lengthscale_constraint = Interval(0.01, 0.5) if use_ard else Interval(0.01, 2.5)

        if self.kern == 'transformed_overlap':
            base_kernel = TransformedCategorical(
                lengthscale_constraint=lengthscale_constraint,
                ard_num_dims=input_dim if use_ard else None
            )
        else:
            raise ValueError
        return ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        # 分离设计变量和保真度 (假设最后一个维度是保真度)
        design_vars = x[:, :-1]
        fidelity = self.fidelity_transform(x[:, -1]).view(-1)

        # 动态训练高保真GP（如果尚未训练且存在高保真数据）
        if self.high_fidelity_gp is None:
            # 使用原始训练数据来确定高保真数据点
            hf_mask = (self.train_x[:, -1] > 0.5)  # 保真度>0.5视为高保真
            hf_x = self.train_x[hf_mask, :-1]  # 只取设计变量
            hf_y = self.train_y[hf_mask]

            if len(hf_y) > 1:  # 至少需要2个点才能训练GP
                self.high_fidelity_gp = train_gp(
                    hf_x, hf_y.squeeze(-1),
                    use_ard=True,
                    num_steps=100,
                    kern=self.kern,
                )

        # 混合预测
        # 获取低保真预测
        lf_pred = self.low_fidelity_model(design_vars).view(-1)  # [batch_size]
        if self.high_fidelity_gp is not None:
            hf_pred = self.high_fidelity_gp(design_vars).mean.view(-1)  # [batch_size]
            mixed_mean = fidelity * hf_pred + (1 - fidelity) * lf_pred  # 都是[batch_size]
        else:
            mixed_mean = lf_pred.view(-1)

        covar_x = self.covar_module(x)  # 使用完整输入（包含保真度）
        return MultivariateNormal(mixed_mean, covar_x)


# GP Model
class GP(ExactGP): # 这个是主要的surrogate model，会在train_gp里被调用。是gpytorch里面ExactGP的子类
    def __init__(self, train_x, train_y, kern, likelihood,
                 outputscale_constraint, ard_dims, cat_dims=None,
                 prior=False, t_mat=None, frac_j=None):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        if prior:
            self.mean_module =  PMedianMean(t_mat, frac_j)
        else:
            self.mean_module = ConstantMean()  # 常数mean, 指的是prior是constant
        self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint)   # specify kernel是什么

    def forward(self, x):  # 这个是标准的GP
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)  # , cat_dims, int_dims)
        return MultivariateNormal(mean_x, covar_x)  # 返回的是这个分布。其中的kernel参数会逐步学习更新


def train_gp(train_x, train_y, use_ard, num_steps, kern='transformed_overlap', hypers={},  # 单独的函数，用来训练GP
             cat_dims=None, cont_dims=None,
             int_constrained_dims=None,
             noise_variance=None,
             cat_configs=None,
             mf=False,
             prior=False, t_mat=None, frac_j=None,
             **params):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.  # 目前这里写的y需要standardize
    （train_x, train_y）: pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    int_constrained_dims: **Of the continuous dimensions**, which ones additionally are constrained to have integer
        values only?
    """
    assert train_x.ndim == 2 # 有n有x的dim
    assert train_y.ndim == 1 # 应该就是一个向量，列了y
    assert train_x.shape[0] == train_y.shape[0]  # 这里应该是指长度要一样

    # Create hyper parameter bounds # 这里都是在创立变量并设定constraint。对于noise的constraint并没有什么大用
    if noise_variance is None:  # 好像模型默认是None。如果没写noise_variance。就用0.005
        noise_variance = 0.005
        noise_constraint = Interval(1e-6, 0.1)      # 这个是指的noise的方差有个constraint。用在likelihood里面，要去估计。
    else:
        if np.abs(noise_variance) < 1e-6: # 如果很小用0.05代替
            noise_variance = 0.05
            noise_constraint = Interval(1e-6, 0.1)
        else:  # 否则就按给的来
            noise_constraint = Interval(0.99 * noise_variance, 1.01 * noise_variance) 
    if use_ard:  # 判定是否是ard。如果ard，则每个lengthscale会不一样，否则lengthscale一样。
        lengthscale_constraint = Interval(0.01, 0.5)  # lengthscale是kernel里面除的那个l，是一个参数
    else:
        lengthscale_constraint = Interval(0.01, 2.5)  # [0.005, sqrt(dim)]
    # outputscale_constraint = Interval(0.05, 20.0)
    outputscale_constraint = Interval(0.5, 5.)

    # Create models  # 调用gpytorch的包，The standard likelihood for regression. Assumes a standard homoskedastic noise model
    # 它是_GaussianLikelihoodBase的子类，而这个是Likelihood的子类 -> _Likelihood -> Module -> nn.Module
    # 里面调用了HomoskedasticNoise()的function。
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)  # 这个还加了一个to device
    # train_x = onehot2ordinal(train_x, cat_dims)
    ard_dims = train_x.shape[1] if use_ard else None        # 如果ard则dim要是d，否则是None就行

    # 下面这里一长串就是说kernel是什么
    if kern == 'overlap':
        kernel = CategoricalOverlap(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'transformed_overlap': # 我们基本上只用这个和上面的一个。
        kernel = TransformedCategorical(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'diffusion_kernel':
        kernel = DiffusionKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, categories = [2 for i in range(train_x.shape[1])], )
    elif kern == 'combined_kernel':
        kernel = CombinedKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims+1, categories = [2 for i in range(train_x.shape[1])], )
    elif kern == 'ordinal':
        kernel = OrdinalKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, config=cat_configs)
    elif kern == 'mixed':
        assert cat_dims is not None and cont_dims is not None, 'cat_dims and cont_dims need to be specified if you wish' \
                                                               'to use the mix kernel'
        kernel = MixtureKernel(cat_dims, cont_dims,
                               categorical_ard=use_ard, continuous_ard=use_ard,
                               integer_dims=int_constrained_dims,
                               **params)
    elif kern == 'mixed_overlap':
        kernel = MixtureKernel(cat_dims, cont_dims,
                               categorical_ard=use_ard, continuous_ard=use_ard,
                               categorical_kern_type='overlap',
                               integer_dims=int_constrained_dims,
                               **params)
    else:
        raise ValueError('Unknown kernel choice %s' % kern)

    ########## 这里是主要的GP model #########
    if mf:
        assert t_mat is not None and frac_j is not None, "MF model requires t_mat and frac_j"

        model = MultiFidelityGP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kern=kern,
            low_fidelity_model=lambda x: PMedianMean(t_mat, frac_j)(x),
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)
    else:
        model = GP(     # 这个是主要的surrogate model。是gpytorch -> ExactGP -> GP (空壳) -> Module -> nn.Module。
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kern=kernel,
            # lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
            prior=prior,
            t_mat=t_mat,
            frac_j=frac_j
        ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train() # 这两句知识声明开始进入训练状态
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model) # 损失函数。给mll放进数据之后可以直接.backward()

    # Initialize model hypers
    if hypers:   # 目前默认的是初始为空的，所以不进入这里
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)  # 这里应该是定义了lengthscale的初始化参数
        hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
        model.initialize(**hypers)   # 这里应该是更新这些超参的地方

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)
    for _ in range(num_steps): # 这个主要是训练模型里参数的，更新模型
        optimizer.zero_grad()
        output = model(train_x, )   # 返回的应该是 MultivariateNormal 分布
        loss = -mll(output, train_y).float()  # 感觉是train_y和这个分布的-mll。
        loss.backward()
        optimizer.step()
    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):  # initialize初始点用的
    import time # 没用
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))   # 感觉像是对于hypercube上，一共有多少个点和一共有多少dim
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)  # 这里是一个小于1的数
    random.seed(random.randint(0, 1e6))
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def compute_hamming_dist(x1, x2, categorical_dims, normalize=False): # 算hamming distance。看看我们是否可以直接用这个。
    """
    Compute the hamming distance of two one-hot encoded strings.
    :param x1:
    :param x2:
    :param categorical_dims: list of lists. e.g.
    [[1, 2], [3, 4, 5, 6]] where idx 1 and 2 correspond to the first variable, and
    3, 4, 5, 6 coresponds to the second variable with 4 possible options
    :return:
    """
    dist = 0.
    for i, var_groups in enumerate(categorical_dims):
        if not np.all(x1[var_groups] == x2[var_groups]):
            dist += 1.
    if normalize:
        dist /= len(categorical_dims)
    return dist


def compute_hamming_dist_ordinal(x1, x2, n_categories=None, normalize=False): # 这个就是计算hamming distance是多少。normalize不需要
    """Same as above, but on ordinal representations."""
    hamming = (x1 != x2).sum()   
    if normalize:
        return hamming / len(x1)
    return hamming


def sample_neighbour(x, categorical_dims):
    """Sample a neighbour (i.e. of unnormalised Hamming distance of 1) from x"""
    x_pert = deepcopy(x)
    # Sample a variable where x_pert will differ from the selected sample
    random.seed(random.randint(0, 1e6))
    choice = random.randint(0, len(categorical_dims) - 1)
    # Change the value of that variable randomly
    var_group = categorical_dims[choice]
    # Confirm what is value of the selected variable in x (we will not sample this point again)
    for var in var_group:
        if x_pert[var] != 0:
            break
    value_choice = random.choice(var_group)
    while value_choice == var:
        value_choice = random.choice(var_group)
    x_pert[var] = 0
    x_pert[value_choice] = 1
    return x_pert


def sample_neighbour_ordinal(x, n_categories):  # 这个是用到sample neighbour的，然后再拿去判断是否符合hamming。但是按照这个情况感觉每次只改一个肯定符合啊。
    # 感觉这个是我们要改的，因为我们有constraint的话需要一个0->1 和一个 1->0。
    """Same as above, but the variables are represented ordinally."""
    x_pert = deepcopy(x)
    # Chooose a variable to modify
    choice = random.randint(0, len(n_categories) - 1)               # 随机选一个x的dim
    # Obtain the current value.
    curr_val = x[choice]                                            # 看看那个现在的值
    options = [i for i in range(n_categories[choice]) if i != curr_val]     # 感觉是从所有可行的里面随机选一个category的值
    x_pert[choice] = random.choice(options)                         # 把那个choice那一位的换成一个random的choice
    return x_pert

def sample_neighbour_ordinal_binary_constraint(x, n_categories, hamming):  # 这个是我加的
    # hamming是当前符合条件的hamming distance TR
    x_pert = deepcopy(x)
    # 跑几遍neighbour
    n_run = 2 if hamming > 2 else 1
    np.random.seed(random.randint(0, 1e6))
    for _ in range(n_run):
        # permute id
        id_permut = np.random.permutation(np.arange(len(x_pert)))
        # Chooose a variable to swap
        choice = id_permut[0]
        # Obtain the current value.
        curr_val = x_pert[choice]
        # 找到那个需要swap且与当前值不一样的那个
        swap_id = 1
        swap = id_permut[swap_id]
        #pre_value = sum(x_pert)
        while x_pert[swap] == curr_val:
            swap_id += 1
            swap = id_permut[swap_id]
        x_pert[choice], x_pert[swap] = x_pert[swap], x_pert[choice]
        #post_value = sum(x_pert)
        #if (pre_value != post_value):
        #    assert()
    return x_pert

def random_sample_within_discrete_tr(x_center, max_hamming_dist, categorical_dims,
                                     mode='ordinal'):
    """Randomly sample a point within the discrete trust region"""
    if max_hamming_dist < 1:  # Normalised hamming distance is used
        bit_change = int(max_hamming_dist * len(categorical_dims))
    else:  # Hamming distance is not normalized
        max_hamming_dist = min(max_hamming_dist, len(categorical_dims))
        bit_change = int(max_hamming_dist)

    x_pert = deepcopy(x_center)
    # Randomly sample n bits to change.
    modified_bits = random.sample(range(len(categorical_dims)), bit_change)
    for bit in modified_bits:
        n_values = len(categorical_dims[bit])
        # Change this value
        selected_value = random.choice(range(n_values))
        # Change to one-hot encoding
        substitute_values = np.array([1 if i == selected_value else 0 for i in range(n_values)])
        x_pert[categorical_dims[bit]] = substitute_values
    return x_pert


def random_sample_within_discrete_tr_ordinal(x_center, max_hamming_dist, n_categories): # 这个是在tr之内随机选点
    """Same as above, but here we assume a ordinal representation of the categorical variables."""
    random.seed(random.randint(0, 1e6))
    if max_hamming_dist < 1:  
        bit_change = int(max(max_hamming_dist * len(n_categories), 1))  # 这个应该是有几位需要改变，在这个情况下最多是1
    else:
        bit_change = int(min(max_hamming_dist, len(n_categories)))
    x_pert = deepcopy(x_center)
    modified_bits = random.sample(range(len(n_categories)), bit_change) # 随机那些需要被变的bits。目前感觉是全都要改的意思。
    for bit in modified_bits:
        options = np.arange(n_categories[bit])
        x_pert[bit] = int(random.choice(options))                       
    return x_pert


def local_search(x_center, f: Callable,             # 这个很重要
                 config,
                 max_hamming_dist,
                 n_restart: int = 1,  # 我们这个package里好像喂进来的是3
                 batch_size: int = 1,  
                 step: int = 100):
    """
    Local search algorithm
    :param n_restart: number of restarts                                                        
    :param config:
    :param x0: the initial point to start the search
    :param x_center: the center of the trust region. In this case, this should be the optimum encountered so far.
    :param f: the function handle to evaluate x on (the acquisition function, in this case)     # ei或者ucb
    :param max_hamming_dist: maximum Hamming distance from x_center                             # 我们这个问题好像是20？ 或者初始化为20
    :param step: number of maximum local search steps the algorithm is allowed to take.
    :return:
    """

    def _ls(hamming):       # 这个是主要的local search函数。我们应该是在这里做文章，怎么找到合适的更好的sample。
        """One restart of local search"""
        # x0 = deepcopy(x_center)
        x_center_local = deepcopy(x_center)
        tol = 100
        trajectory = np.array([x_center_local])
        x = x_center_local

        acq_x = f(x).detach().numpy() # 这里会读取acquisition function的值

        # 这个for loop是为了得到更好的x周围更好的点
        for i in range(step):
            tol_ = tol
            is_valid = False

            # 这个while loop里是为了读到一个可行的且新的neighbour
            while not is_valid:   # 找到一个valid的neighbour就退出。valid指的是满足hamming条件且之前没见过的。
                neighbour = sample_neighbour_ordinal_binary_constraint(x, config, hamming)         # 这里换过了
                # 如果这个sample出来的是满足hamming distance最大距离且不等于0，且没见过的就返回true，否则tol_ -1之后接着sample一个neighbour
                if 0 < compute_hamming_dist_ordinal(x_center_local, neighbour, config) <= hamming \
                        and not any(np.equal(trajectory, neighbour).all(1)):  # 这里trajectory并没有用到，到后来其实只是比跟初始的x_center一不一样
                    is_valid = True
                else:
                    tol_ -= 1
            if tol_ < 0:        # 如果100次之后还没找到合适的neighbour那就不找了，把原来的center给回来，退出这个function
                logging.info("Tolerance exhausted on this local search thread.")
                return x, acq_x

            # 把x和它neighbour的acq都记下来
            acq_x = f(x).detach().numpy()                              
            acq_neighbour = f(neighbour).detach().numpy()
            # print(acq_x, acq_neighbour)

            # 如果neighbour的acq更好，用logging记下来，并更新x为这个更好的
            if acq_neighbour > acq_x:
                # print('x', x, acq_x)
                # print('neighbour', neighbour, acq_neighbour)
                logging.info(''.join([str(int(i)) for i in neighbour.flatten()]) + ' ' + str(acq_neighbour))
                x = deepcopy(neighbour)   # 这个center是会变的
                # trajectory = np.vstack((trajectory, deepcopy(x)))
        logging.info('local search thread ended with highest acquisition %s' % acq_x)
        # print(compute_hamming_dist_ordinal(x_center, x, n_categories), acq_x)
        # print(x_center)
        return x, acq_x  # 最后返回的是200步内找到的最好的那个

    # 这里开始
    X = []
    fX = []
    for i in range(n_restart):          # 对于我们这个问题一共跑三次。相当于sample 3个，然后后面会找个最好的
        # print('round', i)
        res = _ls(max_hamming_dist)     # 这里就会给到我们x和f(x)，然后加到list里
        X.append(res[0])
        fX.append(res[1])
        # print(res_mercer)

    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]  # 把最好的那些的index拿出来。对于我们问题只拿出来一个
    # select the top-k smallest
    # top_idices = np.argpartition(np.array(fX).flatten(), batch_size)[:batch_size]
    # print(np.array(fX).flatten()[top_idices])
    return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]  # 然后返回


def interleaved_search(x_center, f: Callable,
                       cat_dims,
                       cont_dims,
                       config,
                       ub,
                       lb,
                       max_hamming_dist,
                       n_restart: int = 1,
                       batch_size: int = 1,
                       interval: int = 1,
                       step: int = 200):
    """
    Interleaved search combining both first-order gradient-based method on the continuous variables and the local search
    for the categorical variables.
    Parameters
    ----------
    x_center: the starting point of the search
    cat_dims: the indices of the categorical dimensions
    cont_dims: the indices of the continuous dimensions
    f: function handle (normally this should be the acquisition function)
    config: the config for the categorical variables
    lb: lower bounds (trust region boundary) for the continuous variables
    ub: upper bounds (trust region boundary) for the continuous variables
    max_hamming_dist: maximum hamming distance boundary (for the categorical variables)
    n_restart: number of restarts of the optimisaiton
    batch_size:
    interval: number of steps to switch over (to start with, we optimise with n_interval steps on the continuous
        variables via a first-order optimiser, then we switch to categorical variables (with the continuous ones fixed)
        and etc.
    step: maximum number of search allowed.

    Returns
    -------

    """
    # todo: the batch setting needs to be changed. For the continuous dimensions, we cannot simply do top-n indices.

    from torch.quasirandom import SobolEngine
    from scipy.optimize import minimize, Bounds

    # select the initialising points for both the continuous and categorical variables and then hstack them together
    # x0_cat = np.array([deepcopy(sample_neighbour_ordinal(x_center[cat_dims], config)) for _ in range(n_restart)])
    x0_cat = np.array([deepcopy(random_sample_within_discrete_tr_ordinal(x_center[cat_dims], max_hamming_dist, config))
                       for _ in range(n_restart)])
    # x0_cat = np.array([deepcopy(x_center[cat_dims]) for _ in range(n_restart)])
    seed = np.random.randint(int(1e6))
    sobol = SobolEngine(len(cont_dims), scramble=True, seed=seed)
    x0_cont = sobol.draw(n_restart).cpu().detach().numpy()
    x0_cont = lb + (ub - lb) * x0_cont
    x0 = np.hstack((x0_cat, x0_cont))
    tol = 100
    lb, ub = torch.tensor(lb, dtype=torch.float32), torch.tensor(ub, dtype=torch.float32)

    def _interleaved_search(x0):
        x = deepcopy(x0)
        acq_x = f(x).detach().numpy()
        x_cat, x_cont = x[cat_dims], x[cont_dims]
        n_step = 0
        while n_step <= step:
            # First optimise the continuous part, freezing the categorical part
            def f_cont(x_cont_):
                """The function handle for continuous optimisation"""
                x_ = torch.cat((x_cat_torch, x_cont_)).float()
                return -f(x_)

            x_cont_torch = torch.tensor(x_cont, dtype=torch.float32).requires_grad_(True)
            x_cat_torch = torch.tensor(x_cat, dtype=torch.float32)
            optimizer = torch.optim.Adam([{"params": x_cont_torch}], lr=0.1)
            for _ in range(interval):
                optimizer.zero_grad()
                acq = f_cont(x_cont_torch).float()
                try:
                    acq.backward()
                    # print(x_cont_torch, acq, x_cont_torch.grad)
                    optimizer.step()
                except RuntimeError:
                    print('Exception occured during backpropagation. NaN encountered?')
                    pass
                with torch.no_grad():
                    # Ugly way to do clipping
                    x_cont_torch.data = torch.max(torch.min(x_cont_torch, ub), lb)

            x_cont = x_cont_torch.detach().numpy()
            del x_cont_torch

            # Then freeze the continuous part and optimise the categorical part
            for j in range(interval):
                is_valid = False
                tol_ = tol
                while not is_valid:
                    neighbour = sample_neighbour_ordinal(x_cat, config)
                    if 0 <= compute_hamming_dist_ordinal(x_center[cat_dims], neighbour, config) <= max_hamming_dist:
                        is_valid = True
                    else:
                        tol_ -= 1
                if tol_ < 0:
                    logging.info("Tolerance exhausted on this local search thread.")
                    break
                # acq_x = f(np.hstack((x_cat, x_cont))).detach().numpy()
                acq_neighbour = f(np.hstack((neighbour, x_cont))).detach().numpy()
                if acq_neighbour > acq_x:
                    x_cat = deepcopy(neighbour)
                    acq_x = acq_neighbour
            # print(x_cat, x_cont, acq_x)
            n_step += interval

        x = np.hstack((x_cat, x_cont))
        return x, acq_x

    X, fX = [], []
    for i in range(n_restart):
        res = _interleaved_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])
    top_idices = np.argpartition(np.array(fX).flatten(), -batch_size)[-batch_size:]
    return np.array([x for i, x in enumerate(X) if i in top_idices]), np.array(fX).flatten()[top_idices]
