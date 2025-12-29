# Implementation of various kernels

from typing import Optional
from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel # 这些Kernel也都是写好的
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.constraints import Interval
import torch
import numpy as np


class MixtureKernel(Kernel): # Mixture情况采用的Kernel，我们不需要
    """
    The implementation of the mixed categorical and continuous kernel first proposed in CoCaBO, but re-implemented
    in gpytorch.

    Note that gpytorch uses the pytorch autodiff engine, and there is no need to manually define the derivatives of
    the kernel hyperparameters w.r.t the log-marinal likelihood as in the gpy implementation.
    """
    has_lengthscale = True

    def __init__(self, categorical_dims,
                 continuous_dims,
                 integer_dims=None,
                 lamda=0.5,
                 categorical_kern_type='transformed_overlap',
                 continuous_kern_type='mat52',
                 categorical_lengthscale_constraint=None,
                 continuous_lengthscale_constraint=None,
                 categorical_ard=True,
                 continuous_ard=True,
                 **kwargs):
        """

        Parameters
        ----------
        categorical_dims: the dimension indices that are categorical/discrete
        continuous_dims: the dimension indices that are continuous # 对我们来说这个应该是0
        integer_dims: the **continuous indices** that additionally require integer constraint. # continuous里面是否需要整数。我们用不到
        lamda: \in [0, 1]. The trade-off between product and additive kernels. If this argument is not supplied, then
            lambda will be optimised as if it is an additional kernel hyperparameter
        categorical_kern_type: 'overlap', 'type2'
        continuous_kern_type: 'rbf' or 'mat52' (Matern 5/2)
        categorical_lengthscale_constraint: if supplied, the constraint on the lengthscale of the categorical kernel
        continuous_lengthscale_constraint: if supplied the constraint on the lengthscale of the continuous kernel
        categorical_ard: bool: whether to use Automatic Relevance Determination (ARD) for categorical dimensions
        continuous_ard: bool: whether to use ARD for continouous dimensions
        kwargs: additional parameters.
        """
        super(MixtureKernel, self).__init__(has_lengthscale=True, **kwargs)
        self.optimize_lamda = lamda is None  # 这个出来是个bool，如果lambda没给定则自己optimize一个，注册到nn.parameter里去优化
        self.fixed_lamda = lamda if not self.optimize_lamda else None # 否则，则lambda是固定的值。
        self.categorical_dims = categorical_dims
        self.continuous_dims = continuous_dims
        if integer_dims is not None: # 这个我们用不到。
            integer_dims_np = np.asarray(integer_dims).flatten()
            cont_dims_np = np.asarray(self.continuous_dims).flatten()
            if not np.all(np.in1d(integer_dims_np, cont_dims_np)):
                raise ValueError("if supplied, all continuous dimensions with integer constraint must be themselves "
                                 "contained in the continuous_dimensions!")
            # Convert the integer dims in terms of indices of the continous dims
            integer_dims = np.where(np.in1d(self.continuous_dims, integer_dims))[0]

        self.register_parameter(name='raw_lamda', parameter=torch.nn.Parameter(torch.ones(1)))  # 继承的是nn.Module。就是self._parameters[name] = param
        # The lambda must be between 0 and 1.
        self.register_constraint('raw_lamda', Interval(0., 1.)) # paramter设定一个限制。用了这个之后会限制自动变成raw_lambda_constraint

        # Initialise the kernel # 用什么kernel
        if categorical_kern_type == 'overlap':
            self.categorical_kern = CategoricalOverlap(lengthscale_constraint=categorical_lengthscale_constraint,
                                                       ard_num_dims=len(categorical_dims) if categorical_ard else None)
        elif categorical_kern_type == 'transformed_overlap':
            self.categorical_kern = TransformedCategorical(lengthscale_constraint=categorical_lengthscale_constraint,
                                                           ard_num_dims=len(
                                                                categorical_dims) if categorical_ard else None)
        else:
            raise NotImplementedError("categorical kernel type %s is not implemented. " % categorical_kern_type)

        # By default, we use the Matern 5/2 kernel # 目前我们这里用不到
        if continuous_kern_type == 'mat52': 
            self.continuous_kern = WrappedMatern(nu=2.5, ard_num_dims=len(continuous_dims) if continuous_ard else None,
                                                 integer_dims=integer_dims,
                                                 lengthscale_constraint=continuous_lengthscale_constraint)
        elif continuous_kern_type == 'rbf':
            self.continuous_kern = WrappedRBF(ard_num_dims=len(continuous_dims) if continuous_ard else None,
                                              integer_dims=integer_dims,
                                              lengthscale_constraint=continuous_lengthscale_constraint)
        else:
            raise NotImplementedError("continuous kernel type %s is not implemented. " % continuous_kern_type)

    @property
    def lamda(self):
        if self.optimize_lamda:
            return self.raw_lamda_constraint.transform(self.raw_lamda) # 把求出来的raw_lambda再通过constraint变换一下
        else:
            return self.fixed_lamda

    @lamda.setter
    def lamda(self, value):
        self._set_lamda(value)

    def _set_lamda(self, value):
        if self.optimize_lamda: # 如果不指定，要去优化
            if not isinstance(value, torch.Tensor): # 如果不是tensor就把它变成tensor
                value = torch.as_tensor(value).to(self.raw_lamda)
            self.initialize(raw_lamda=self.raw_lamda_constraint.inverse_transform(value)) # 初始化raw_lambda
        else:
            # Manually restrict the value of lamda between 0 and 1.
            if value <= 0:
                self.fixed_lamda = 0.
            elif value >= 1:
                self.fixed_lamda = 1.
            else:
                self.fixed_lamda = value

    def forward(self, x1, x2, diag=False,
                x1_cont=None, x2_cont=None, **params):
        """
        Note that here I also give options to pass the categorical and continuous inputs separately (instead of jointly)
        because the categorical dimensions will not be differentiable, and thus there would be problems when we optimize
        the acquisition function. # 这里是可以把x的离散的和连续分开来传，也可以一起传。我们的情况下x1_cont和x2_cont一定是None。

        When passed separately, x1 and x2 refer the categorical (non-differentiable) data, whereas x1_cont and x2_cont
        are the continuous (differentiable) data.
        Parameters
        ----------
        x1
        x2
        diag
        x1_cont
        x2_cont
        params

        Returns
        -------

        """
        if x1_cont is None and x2_cont is None:  
            assert x1.shape[1] == len(self.categorical_dims) + len(self.continuous_dims), \
                'dimension mismatch. Expected number of dimensions %d but got %d in x1' % \
                (len(self.categorical_dims) + len(self.continuous_dims), x1.shape[1])
            x1_cont, x2_cont = x1[:, self.continuous_dims], x2[:, self.continuous_dims] # 我们不需要这个
            # the categorical kernels are not differentiable w.r.t inputs, detach them to ensure the computing graph of
            # the autodiff engine is not broken.
            x1_cat, x2_cat = x1[:, self.categorical_dims].detach(), x2[:, self.categorical_dims].detach() # x_1和x_2取出来不要求导。
        else:
            assert x1.shape[1] == len(self.categorical_dims)
            assert x1_cont.shape[1] == len(self.continuous_dims)
            x1_cat, x2_cat = x1, x2
        # same in cocabo.
        return (1. - self.lamda) * (self.categorical_kern.forward(x1_cat, x2_cat, diag, **params) +
                                    self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)) + \
               self.lamda * self.categorical_kern.forward(x1_cat, x2_cat, diag, **params) * \
               self.continuous_kern.forward(x1_cont, x2_cont, diag, **params)


def wrap(x1, x2, integer_dims):  # wrapping 变换
    """The wrapping transformation for integer dimensions according to Garrido-Merchán and Hernández-Lobato (2020)."""
    if integer_dims is not None:
        for i in integer_dims:
            x1[:, i] = torch.round(x1[:, i])
            x2[:, i] = torch.round(x2[:, i])
    return x1, x2


class WrappedMatern(MaternKernel): # MaternKernel. 
    """Matern kernels wrapped integer type of inputs according to
    Garrido-Merchán and Hernández-Lobato in
    "Dealing with Categorical and Integer-valued Variables in Bayesian Optimization with Gaussian Processes"

    Note: we deal with the categorical-valued variables using the kernels specifically used to deal with
    categorical variables (instead of the one-hot transformation).
    """

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedMatern, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class WrappedRBF(RBFKernel, WrappedMatern): # RBF_Kernel. 
    """Similar to above, but applied to RBF."""

    def __init__(self, integer_dims=None, **kwargs):
        super(WrappedRBF, self).__init__(**kwargs)
        self.integer_dims = integer_dims

    def forward(self, x1, x2, diag=False, **params):
        x1, x2 = wrap(x1, x2, self.integer_dims)
        return super().forward(x1, x2, diag=diag, **params)


class CategoricalOverlap(Kernel): # 这个应该Wan2021用来修改的kernel，目前应该也用不到。
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True          # 从Kernel里得知，这里指的是 Θ is a constant diagonal matrix
    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()


class TransformedCategorical(CategoricalOverlap): # 这个应该是我们要用的kernel
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)   # ard是automatic relevance determination
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD      # lambda_i就是文章里面的l_i。n就是文章里的d_h。
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        diff = x1[:, None] - x2[None, :] # 两个的diff，用它来判断哪些hamming是1。
        # print('diff:', diff, ' x1:', x1, ' x1_none:', x1[:, None], ' x2:', x2, ' x2_none:', x2[None, :])
        diff[torch.abs(diff) > 1e-5] = 1 # 令那些差大于1e-5的都为1
        diff1 = torch.logical_not(diff).float() # 只有0变成1，其他都变成0。想当于一样的取出来，否则都不要。

        # print('self.lengthscale: ', self.lengthscale)

        def rbf(d, ard): # 用的是这个
            if ard: # lengthscale不同的情况
                return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale)) # 这里面lengthscale是不同的值。是l_i。
            else:
                return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard): # 这个没implement用不了
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).float()

        # print('k_cat.float():', k_cat.float())
        return k_cat.float()


class OrdinalKernel(Kernel): # 这个kernel我们目前应该用不到
    """
    The ordinal version of TransformedCategorical2 kernel (replace the Kronecker delta with
    the distance metric).
    config: the number of vertices per dimension
    """
    def __init__(self, config, **kwargs):
        super(OrdinalKernel, self).__init__(has_lengthscale=True, **kwargs)
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config).view(-1)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # expected x1 and x2 are of shape N x D respectively
        diff = (x1[:, None] - x2[None, :]) / self.config
        dist = 1. - torch.abs(diff)
        print(dist)
        print(self.lengthscale)
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.exp(
                torch.sum(
                    dist * self.lengthscale, dim=-1
                ) / torch.sum(self.lengthscale)
            )
        else:
            k_cat = torch.exp(
                self.lengthscale * torch.sum(dist, dim=-1) / x1.shape[1]
            )
        if diag:
            return torch.diag(k_cat).float()
        return k_cat.float()



class DiffusionKernel(Kernel):
    r"""
        Computes diffusion kernel over discrete spaces with arbitrary number of categories. 
        Input type: n dimensional discrete input with c_i possible categories/choices for each dimension i 
        As an example, binary {0,1} combinatorial space corresponds to c_i = 2 for each dimension i
        References:
        - https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf (Section 4.4)
        - https://arxiv.org/abs/1902.00448
        - https://arxiv.org/abs/2012.07762
        
        Args:
        :attr:`categories`(tensor, list):
            array with number of possible categories in each dimension            
    """
    has_lengthscale = True
    def __init__(self, categories, **kwargs):
        if categories is None:
            raise ValueError("Can't create a diffusion kernel without number of categories. Please define them!")
        super().__init__(**kwargs)
        self.cats = categories

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)        

        # print('x1.shape[-1]: ', x1.shape[-1])
        # print('self.lengthscale: ', self.lengthscale)

        if diag:
            res = 1.
            for i in range(x1.shape[-1]):
                diff = (x1[..., i] - x2[..., i])
                diff[torch.abs(diff) > 1e-5] = 1
                res *= ((1 - torch.exp(-self.lengthscale[..., i] * self.cats[i]))/(1 + (self.cats[i] - 1) * torch.exp(-self.lengthscale[..., i]*self.cats[i]))).unsqueeze(-1) ** (diff[:, 0, ...])
            return res

        res = 1.
        for i in range(x1.shape[-1]): 
            diff = x1[..., i].unsqueeze(-2)[..., None] - x2[..., i].unsqueeze(-2)
            diff[torch.abs(diff) > 1e-5] = 1
            res *= ((1 - torch.exp(-self.lengthscale[..., i] * self.cats[i]))/(1 + (self.cats[i] - 1) * torch.exp(-self.lengthscale[..., i]*self.cats[i]))).unsqueeze(-1) ** (diff[0, ...])
        # print('res_mercer: ', res_mercer)
        return res


class CombinedKernel(Kernel):
    r"""
        Computes diffusion kernel over discrete spaces with arbitrary number of categories. 
        Input type: n dimensional discrete input with c_i possible categories/choices for each dimension i 
        As an example, binary {0,1} combinatorial space corresponds to c_i = 2 for each dimension i
        References:
        - https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf (Section 4.4)
        - https://arxiv.org/abs/1902.00448
        - https://arxiv.org/abs/2012.07762
        
        Args:
        :attr:`categories`(tensor, list):
            array with number of possible categories in each dimension            
    """
    has_lengthscale = True
    def __init__(self, categories, **kwargs):
        if categories is None:
            raise ValueError("Can't create a diffusion kernel without number of categories. Please define them!")
        super().__init__(**kwargs)
        self.cats = categories

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        # Diffusion Kernel
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)        

        if diag:
            res = 1.
            for i in range(x1.shape[-1]):
                diff = (x1[..., i] - x2[..., i])
                diff[torch.abs(diff) > 1e-5] = 1
                res *= ((1 - torch.exp(-self.lengthscale[...,0] * self.cats[i]))/(1 + (self.cats[i] - 1) * torch.exp(-self.lengthscale[...,0]*self.cats[i]))).unsqueeze(-1) ** (diff[:, 0, ...])
            return res

        res = 1.
        for i in range(x1.shape[-1]): 
            diff = x1[..., i].unsqueeze(-2)[..., None] - x2[..., i].unsqueeze(-2)
            diff[torch.abs(diff) > 1e-5] = 1
            res *= ((1 - torch.exp(-self.lengthscale[...,0] * self.cats[i]))/(1 + (self.cats[i] - 1) * torch.exp(-self.lengthscale[...,0]*self.cats[i]))).unsqueeze(-1) ** (diff[0, ...])
        # print('res_mercer: ', res_mercer)

        # print('self.lengthscale[...,0]',self.lengthscale[...,0])

        diff = x1[:, None] - x2[None, :] # 两个的diff，用它来判断哪些hamming是1。
        # print('diff:', diff, ' x1:', x1, ' x1_none:', x1[:, None], ' x2:', x2, ' x2_none:', x2[None, :])
        diff[torch.abs(diff) > 1e-5] = 1 # 令那些差大于1e-5的都为1
        diff1 = torch.logical_not(diff).float() # 只有0变成1，其他都变成0。想当于一样的取出来，否则都不要。

        # print('self.lengthscale: ', self.lengthscale)
        # Overlap Kernel
        def rbf(d, ard): # 用的是这个
            if ard: # lengthscale不同的情况
                return torch.exp(torch.sum(d * self.lengthscale[...,1:], dim=-1) / torch.sum(self.lengthscale[...,1:])) # 这里面lengthscale是不同的值。是l_i。
            else:
                return torch.exp(self.lengthscale[...,1:] * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard): # 这个没implement用不了
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).float()

        # print('k_cat.float():', k_cat.float())

        v = torch.add(res, k_cat.float())

        # print('v: ', v)

        return v



if __name__ == '__main__':
    # Test whether the ordinal kernel is doing ok
    import numpy as np
    import matplotlib.pyplot as plt
    x1 = torch.tensor([[13.,  4.],
        [43., 15.],
        [32., 19.],
        [41.,  9.],
        [47., 44.],
        [48., 21.],
        [15., 24.],
        [20., 13.],
        [36., 46.],
        [19., 17.],
        [35.,  6.],
        [39., 50.],
        [24., 10.],
        [45., 18.],
        [29.,  3.],
        [17., 27.],
        [25., 16.],
        [37., 29.],
        [16.,  2.],
        [ 3., 38.]])

    o = OrdinalKernel(config=[51, 51])
    # o.has_lengthscale =  True # 我加的，好像解决之前的问题，但是还有新问题，应该是因为版本造成的
    # o.lengthscale = 1. # 这里目前有问题，可能是因为Gpytorch包的更新，不过无所谓这个我们用不到，只是用来测试的。
    K = o.forward(x1, x1).detach().numpy()
    plt.imshow(K)
    plt.colorbar()
    plt.show()
