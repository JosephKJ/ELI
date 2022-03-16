import torch
from scipy.linalg import null_space
import torch.nn.functional as F


class GFK:
    def __init__(self, dim=20):
        self.dim = dim
        self.eps = 1e-2
        self.source_dim=50
        self.target_dim=50

    def decompose(self, input, subspace_dim=2):
        square_mat = input.transpose(0, 1).mm(input)
        sA_minushalf = self.sqrt_newton_schulz_minus(square_mat, numIters=1)
        ortho_mat = input.double().mm(sA_minushalf)
        ortho_mat = ortho_mat.float()
        return ortho_mat[:,:subspace_dim]

    def train_pca_tall(self, data,  subspace_dim):
        '''
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param mu_data: mu
        :param std_data: std
        :param subspace_dim: dim
        :return: a wrapped machine object
        '''
        data2 = data - data.mean(0)
        uu, ss, vv = torch.svd(data2.float())
        subspace = uu[:, :subspace_dim]
        return subspace

    def sqrt_newton_schulz_autograd(self, A, numIters=2):
        dim = A.data.shape[0]
        normA = A.mul(A).sum(dim=0).sum(dim=0).sqrt()
        Y = A.div(normA.view(1, 1).expand_as(A))
        I = torch.eye(dim,dim).double().cuda()
        Z = torch.eye(dim,dim).double().cuda()

        for i in range(numIters):
            T = 0.5*(3.0*I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
        sA = Y*torch.sqrt(normA).expand_as(A)
        return sA

    def sqrt_newton_schulz_minus(self, A, numIters=1):
        A = A.double()
        dim = A.data.shape[0]
        normA = A.mul(A).sum(dim=0).sum(dim=0).sqrt()
        Y = A.div(normA.view(1, 1).expand_as(A))
        I = torch.eye(dim,dim).double().cuda()
        Z = torch.eye(dim,dim).double().cuda()

        for i in range(numIters):
            T = 0.5*(3.0*I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)

        sZ = Z * 1./torch.sqrt(normA).expand_as(A) ### diabgi karena ini minus power
        return sZ

    def fit(self, input1, input2):
        '''
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return: GFK kernel G
        '''

        input1 = F.normalize(input1, dim=-1, p=2)
        input2 = F.normalize(input2, dim=-1, p=2)

        source_dim = input1.size(0)-2#min(64, input1.size(0)-2)#input1.size(0)//2#-2
        target_dim = input2.size(0)-2#min(64, input2.size(0)-2)#16#input2.size(0)//2#-2
        num_nullspacedim = 60

        ## PSRS #####
        source = input1
        Ps = self.train_pca_tall(source.t(), subspace_dim=source_dim)#.detach()
        Rs = torch.from_numpy(null_space(Ps.t().cpu().detach().numpy())[:, :num_nullspacedim]).cuda()
        Ps = torch.cat([Ps, Rs], dim=1) ### adding columns
        N = Ps.shape[1]  # L = NxK shot - 1

        target = input2
        Pt =  self.train_pca_tall(target.t(), subspace_dim=target_dim)

        G = self.gfk_G(Ps, Pt, N, source_dim, target_dim).detach().float()

        qq1 = input1-input2
        projected_qq = G.t().mm(qq1.t()).t()

        projected_qq_norm = projected_qq
        loss =  torch.sum((qq1) * projected_qq_norm, dim=-1)
        loss_kd = loss.mean()
        return loss_kd

    def gfk_G(self, Ps, Pt, N, source_dim, target_dim):
        A = Ps[:, :source_dim].t().mm(Pt)
        B = Ps[:, source_dim:].t().mm(Pt)

        ######## GPU #############

        UU, SS, VV = self.HOGSVD_fit([A, B])
        V1, V2, V, Gam, Sig = UU[0], UU[1], VV, SS[0], SS[1]
        V2 = -V2

        Gam = Gam.clamp(min=-1., max=1.)
        theta = torch.acos(Gam)

        B1 = torch.diag( 0.5* (1 + (torch.sin(2 * theta) / (2. * theta + 1e-12))))
        B2 = torch.diag( 0.5* (torch.cos(2 * theta) - 1) / (2 * theta + 1e-12))
        B3 = B2
        B4 = torch.diag( 0.5*  (1. - (torch.sin(2. * theta) / (2. * theta + 1e-12))))

        delta1_1 = torch.cat((V1, torch.zeros((N - source_dim, target_dim)).cuda()), dim=0)
        delta1_2 = torch.cat((torch.zeros((source_dim, target_dim)).cuda(), V2), dim=0)

        delta1 = torch.cat((delta1_1, delta1_2), dim=1)


        delta2_1 = torch.cat((B1, B3), dim=0)
        delta2_2 = torch.cat((B2, B4), dim=0)
        delta2 = torch.cat((delta2_1, delta2_2), dim=1)


        delta3_1 = torch.cat((V1.t(), torch.zeros((target_dim, source_dim)).cuda()),dim=0)
        delta3_2 = torch.cat((torch.zeros((target_dim,   N-source_dim)).cuda(), V2.t()),dim=0)
        delta3 = torch.cat((delta3_1, delta3_2), dim=1)

        mm_delta = torch.matmul(delta1, delta2)

        delta = torch.matmul(mm_delta, delta3)
        G = torch.matmul(torch.matmul(Ps, delta), Ps.t()).float()

        return G

    ############################## HOGSVD #########################
    def inverse(self, X):
        eye = torch.diag(torch.randn(X.shape[0]).cuda()).double() * self.eps
        Z = self.sqrt_newton_schulz_minus(X.double(), numIters=1).float()
        A = Z.mm(Z)
        return A.float()

    def HOGSVD_fit_S(self, X):
        N = len(X)
        data_shape = X[0].shape
        A = [torch.matmul(x.transpose(0, 1), x).float().cuda()   for x in X]
        A_inv = [self.inverse(a.double()).float().cuda() for a in A]
        S = torch.zeros((data_shape[1], data_shape[1])).float().cuda()
        for i in range(N):
            for j in range(i + 1, N):
                S = S + (torch.matmul(A[i], A_inv[j]) + torch.matmul(A[j], A_inv[i]))
        S = S / (N * (N - 1))
        return S

    def _eigen_decompostion(self, X, subspace_dim):
        V, eigen_values, V_t = torch.svd(X.double())
        return  V.float()

    def HOGSVD_fit_B(self, X, V):
        X = [x.float().cuda() for x in X]
        V_inv = V.t()
        B = [torch.matmul(V_inv, x.transpose(0, 1)).transpose(0, 1) for x in X]
        return B

    def HOGSVD_fit_U_Sigma(self, B):
        B = [b for b in B]
        sigmas = torch.stack([torch.norm(b, dim=0) for b in B])
        U = [b / (sigma  ) for b, sigma in zip(B, sigmas)]
        return sigmas, U

    def HOGSVD_fit(self, X):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array-like, shape (n_samples, (n_rows_i, n_cols)
            List of training input samples. Eah input element has
            the same numbe of columns but can have unequal number of rows.
        Returns
        -------
        self : object
            Returns self.
        """

        X = [x for x in X]
        # Step 1: Calculate normalized S
        S = self.HOGSVD_fit_S(X).float()
        V = self._eigen_decompostion(S, S.size(0))
        B = self.HOGSVD_fit_B(X, V)
        sigmas, U = self.HOGSVD_fit_U_Sigma(B)
        return U, sigmas, V
