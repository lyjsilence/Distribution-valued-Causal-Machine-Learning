import os
import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.model_selection import KFold
from models import NFR, MLP, Flow_IPW
from baseline_models import est_ECDF, Cross_validate, Regression
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import sim_dataset_cdf, sim_dataset_reg
from training import train_regression, train_IPW
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')
import optimal_h as opth
from scipy.stats import beta, norm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DataGenerator:
    def __init__(self, args):
        self.sample_size = 100000
        self.exp_size = args.sample_size
        self.num_cov = args.num_cov
        self.num_obs = args.num_obs
        self.obs_dim = args.obs_dim
        self.save_path = args.data_save_path
        self.phi = 1
        self.c = 0

        self.basis_list = [self.inv_beta_1, self.inv_beta_2, self.inv_beta_3, self.inv_beta_4, self.inv_beta_5]

    def inv_beta_1(self, alpha):
        return beta.ppf(alpha, 0.1, 0.1)

    def inv_beta_2(self, alpha):
        return beta.ppf(alpha, 0.2, 0.2)

    def inv_beta_3(self, alpha):
        return beta.ppf(alpha, 0.3, 0.3)

    def inv_beta_4(self, alpha):
        return beta.ppf(alpha, 0.4, 0.4)

    def inv_beta_5(self, alpha):
        return beta.ppf(alpha, 0.5, 0.5)

    def non_linear_x_fun(self, x):
        return np.exp(x / self.phi) / np.sum(np.exp(x / self.phi)).reshape(-1, 1)

    def inverse_cdf(self, alpha, i, ground=False, ground_D=0.0, plot=False):
        self.D_mean = self.D.mean()
        x = np.array([self.X[i, 2*j] * self.X[i, 2*j+1] for j in range(int(self.num_cov/2))])
        non_linear_x = self.non_linear_x_fun(x)
        basis = np.stack([self.basis_list[k](alpha) for k in range(len(self.basis_list))])
        sum_basis = np.matmul(non_linear_x, basis)

        if ground:
            obs = self.c + (1-self.c) * (self.D_mean + np.exp(ground_D)) * sum_basis
        else:
            obs = self.c + (1-self.c) * (self.D_mean + np.exp(self.D[i])) * sum_basis \
                  + np.random.normal(loc=0, scale=0.05, size=[1, sum_basis.shape[1]])
            obs_no_noise = self.c + (1-self.c) * (self.D_mean + np.exp(self.D[i])) * sum_basis
        if plot:
            return obs, obs_no_noise
        else:
            return obs

    def create_D(self, X):

        coef_mean = np.array([1/10 for _ in range(10)]).reshape(-1, 1)
        coef_var = np.array([1/20 for _ in range(10)]).reshape(-1, 1)

        mean = np.dot(X, coef_mean)
        var = np.log(1 + np.exp(np.dot(X, coef_var)))

        D = np.zeros([self.sample_size, 1])

        for i in range(self.sample_size):
            D[i] = np.random.normal(loc=mean[i], scale=var[i], size=1)
        return D

    def generate(self):
        if os.path.exists(args.data_save_path):

            self.X = np.array(pd.read_csv(os.path.join(args.data_save_path, 'X.csv'), header=None))
            self.D = np.array(pd.read_csv(os.path.join(args.data_save_path, 'D.csv'), header=None))
            self.Y = np.array(pd.read_csv(os.path.join(args.data_save_path, 'Y.csv'), header=None))

        else:
            # generate covariates by assuming all features follows normal distribution with specific mean and variance
            self.X = np.zeros([self.sample_size, self.num_cov])
            self.X[:, 0:2] = np.random.normal(loc=-2, scale=1, size=[self.sample_size, 2])
            self.X[:, 2:4] = np.random.normal(loc=-1, scale=1, size=[self.sample_size, 2])
            self.X[:, 4:6] = np.random.normal(loc=0, scale=1, size=[self.sample_size, 2])
            self.X[:, 6:8] = np.random.normal(loc=1, scale=1, size=[self.sample_size, 2])
            self.X[:, 8:10] = np.random.normal(loc=2, scale=1, size=[self.sample_size, 2])

            # generate treatment
            self.D = self.create_D(self.X)

            # generate observations by basis function
            y_list = []
            for idx in range(self.sample_size):
                alpha = np.random.uniform(low=0, high=1, size=self.num_obs)
                y_list.append(self.inverse_cdf(alpha, idx).squeeze())
            self.Y = np.stack(y_list)

            # save the data
            os.makedirs(args.data_save_path)
            pd.DataFrame(self.X).to_csv(os.path.join(args.data_save_path, 'X.csv'), index=False, header=False)
            pd.DataFrame(self.D).to_csv(os.path.join(args.data_save_path, 'D.csv'), index=False, header=False)
            pd.DataFrame(self.Y).to_csv(os.path.join(args.data_save_path, 'Y.csv'), index=False, header=False)

        exp_idx = np.random.choice(range(self.sample_size), self.exp_size, replace=False)
        X, Y, D = self.X[exp_idx, :], self.Y[exp_idx, :], self.D[exp_idx, :]

        return X, D, Y

    def plot(self, num_sample):
        set_seed(3)
        sample = np.random.choice(self.sample_size, num_sample, replace=False)
        for idx in sample:
            alpha = np.linspace(0, 1, self.num_obs)
            obs, obs_nonoise = self.inverse_cdf(alpha, idx, plot=True)
            plt.scatter(alpha, obs.squeeze(), s=5, alpha=0.5)
            plt.plot(alpha, obs_nonoise.squeeze(), label=f'instance {idx}')

        plt.xlabel('Quantiles')
        plt.ylabel('$Y^{-1}$')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(args.data_save_path, 'instances.pdf'))
        plt.show()

    def groud_truth(self, quantiles):
        '''Ground truth for y^-1'''

        alpha = quantiles
        y_inv_mean_list = []

        for ground_D in [-0.5, 0.00, 0.5]:
            y_inv_list = []
            for idx in range(self.sample_size):
                y_inv_list.append(self.inverse_cdf(alpha, idx, ground=True, ground_D=ground_D).squeeze())

            y_inv_mean_list.append(np.mean(y_inv_list, axis=0).reshape(1, self.obs_dim))

        print(np.stack(y_inv_mean_list, axis=0).squeeze())
        return np.stack(y_inv_mean_list, axis=0).squeeze()



parser = argparse.ArgumentParser(description='Simulation experiment on causal function')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_save_path', type=str, default='data/Simulation')
parser.add_argument('--results_save_path', type=str, default='results')
parser.add_argument('--dataset', type=str, default='Simulation')

parser.add_argument('--sample_size', type=int, default=10000, help='Number of samples')
parser.add_argument('--num_obs', type=int, default=100, help='Number of observations (y) for each sample')
parser.add_argument('--num_cov', type=int, default=10, help='Number of covariates')


parser.add_argument('--obs_dim', type=int, default=9)
parser.add_argument('--act_fn', type=str, default="relu", choices=["tanh", "softplus", "elu", "relu"])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--K_fold', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--epochs_DR', type=int, default=200)
parser.add_argument('--epochs_IPW', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0)


parser.add_argument('--reg_met', type=str, default='NFR', choices=['MCP', 'lasso', 'ridge', 'elastic net', 'MLP', 'NFR', 'rep_NFR', 'Flow'])
parser.add_argument('--IPW_met', type=str, default='Flow', choices=['Gaussian', 'Flow'])
parser.add_argument('--kernel_met', type=str, default='epanechnikov', choices=['epanechnikov', 'gaussian', 'triweight'])
parser.add_argument('--start_exp', type=int, default=0)
parser.add_argument('--end_exp', type=int, default=100)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()


def compute_MAE(true, pred):
    return np.mean(np.abs(np.array(true) - np.array(pred)))


def compute_RE(true, pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true)))


class Estimator:
    def __init__(self, Y_inv, D, X, Y, quantiles, K_fold, reg_met):
        if reg_met in ['MCP', 'lasso', 'ridge', 'elastic net', 'MLP', 'NFR']:
            assert Y_inv is not None

        elif reg_met == 'Flow':
            assert Y is not None


        self.K_fold = K_fold
        self.quantiles = quantiles
        self.Phi = self.B_spline(quantiles)
        self.N = X.shape[0]

        self.X = X
        self.D = D.reshape(-1, 1)
        self.XD = np.hstack([X, self.D])
        self.Y = Y
        self.Y_inv = Y_inv

        self.conter_D_test = [-0.5, 0.0, 0.5]
        self.reg_met = reg_met
        self.args = args

        self.lambdas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        self.alphas = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]
        self.l1_ratio = [.01, .1, .5, .9, .99]
        self.valid_loss_met = 'mean_square_loss'
        self.kernel_met = args.kernel_met

        if self.reg_met == 'MCP':
            self.lambda_star \
                = Cross_validate(self.XD, self.Y_inv, self.Phi, self.valid_loss_met, lambdas=self.lambdas, reg_met=self.reg_met)
            self.alpha_star, self.l1_ratio_star = None, None

        elif self.reg_met == 'lasso' or reg_met == 'ridge':
            self.alpha_star \
                = Cross_validate(self.XD, self.Y_inv, self.Phi, self.valid_loss_met, alphas=self.alphas, reg_met=self.reg_met)
            self.l1_ratio_star, self.lambda_star = None, None

        elif self.reg_met == 'elastic net':
            self.alpha_star, self.l1_ratio_star \
                = Cross_validate(self.XD, self.Y_inv, self.Phi, self.valid_loss_met,  alphas=self.alphas,
                                 l1_ratio=self.l1_ratio, reg_met=self.reg_met)
            self.lambda_star = None

    def B_spline(self, t):
        from scipy.interpolate import BSpline

        spline_deg = 3
        num_est_basis = len(t) - spline_deg - 1
        Phi = np.zeros((len(t), num_est_basis))
        for k in range(num_est_basis):
            coeff_ = np.eye(num_est_basis)[k, :]
            fun = BSpline(t, coeff_, spline_deg)
            Phi[:, k] = fun(t)
        return Phi

    def DR_estimate(self, XD_train, Y_inv_train, XD_test, Y_inv_test, Y_train, Y_test):
        est_XD_test_list = []
        for d in self.conter_D_test:
            est_XD_test = XD_test.copy()
            est_XD_test[:, -1] = d
            est_XD_test_list.append(est_XD_test)

        if self.reg_met in ['MCP', 'lasso', 'ridge', 'elastic net']:
            self.est_Y_inv_list \
                = Regression(XD_train, XD_test, est_XD_test_list, Y_inv_train, Y_inv_test,
                             self.Phi, self.lambda_star, self.alpha_star, self.l1_ratio_star, self.reg_met)

        elif self.reg_met in ['NFR', 'MLP']:
            if self.reg_met == 'NFR':
                models = NFR(dim_in=args.num_cov+1, dim_out=args.num_cov+1, t=self.quantiles, dim_hidden=[64, 64],
                             activation=args.act_fn, device=device).to(device)
            elif self.reg_met == 'MLP':
                models = MLP(dim_in=args.num_cov+1, dim_out=len(quantiles), device=device).to(device)

            dl_train = DataLoader(dataset=sim_dataset_reg(XD_train, Y_inv_train), shuffle=True, batch_size=args.batch_size)
            dl_test = DataLoader(dataset=sim_dataset_reg(XD_test, Y_inv_test), shuffle=False, batch_size=len(XD_test))

            dl_est_test = []
            for d in range(len(self.conter_D_test)):
                dl_est_test.append(DataLoader(dataset=sim_dataset_reg(est_XD_test_list[d], Y_inv_test), shuffle=False, batch_size=len(est_XD_test_list[d])))

            est_Y_inv_list = train_regression(args, models, dl_train, dl_test, dl_est_test, device=device, verbose=True)

        return est_Y_inv_list


    def GPS_estimate(self, XD_train, XD_test):
        est_XD_test_list = []
        for d in self.conter_D_test:
            est_XD_test = XD_test.copy()
            est_XD_test[:, -1] = d
            est_XD_test_list.append(est_XD_test)

        dl_train = DataLoader(XD_train, shuffle=True, batch_size=args.batch_size)
        dl_test = DataLoader(XD_test, shuffle=False, batch_size=XD_test.shape[0])
        dl_est_test_list = []
        for d in range(len(self.conter_D_test)):
            dl_est_test_list.append(DataLoader(est_XD_test_list[d], shuffle=False, batch_size=len(est_XD_test_list[d])))

        if args.IPW_met == 'Flow':
            args.input_dim = 1
            args.cond_dim = 10
            args.hidden_dim = 32
            args.rademacher = False
            args.flow_time = 1.0
            args.train_T = True
            args.solver = 'dopri5'
            args.num_blocks = 1
            args.batch_norm = True

            args.kinetic_energy = 0.1
            args.jacobian_norm2 = 0.1
            args.directional_penalty = None

            models = Flow_IPW(args, device=device).to(device)

            est_prob_list = train_IPW(args, models, dl_train, dl_test, dl_est_test_list, device=device, verbose=True)

        return est_prob_list


    def Bandwidth_estimate(self, est_pi_list, est_Y_inv_list, XD_test_list, Y_inv_test_list,
                           opt_h_met='opt_h_surface', kernel_met='gaussian'):
        bandwidth_list = []
        for i in range(len(self.conter_D_test)):
            d = self.conter_D_test[i]  # tested virtual treatment
            if opt_h_met == 'opt_h_surface':
                bandwidth = opth.optbandwidth_surface(d, est_pi_list[i], est_Y_inv_list[i], XD_test_list, Y_inv_test_list, kernel_met)
            elif opt_h_met == 'opt_h_quantile':
                bandwidth = opth.optbandwidth(d, est_pi_list[i], est_Y_inv_list[i], XD_test_list, Y_inv_test_list, self.kernel_met)
            bandwidth_list.append(bandwidth)
        return bandwidth_list


    def estimate(self):
        kf = KFold(n_splits=self.K_fold)
        self.est_DR_list = np.zeros([len(self.conter_D_test), args.obs_dim, self.K_fold])
        self.est_IPW_list = np.zeros([len(self.conter_D_test), args.obs_dim, self.K_fold])
        self.est_DML_list = np.zeros([len(self.conter_D_test), args.obs_dim, self.K_fold])
        self.weight_list = []
        iter = 0

        for train_idx, test_idx in kf.split(self.XD):
            print(f'\ncross fitting: {iter}')

            # train-test-split
            self.XD_train, self.XD_test = self.XD[train_idx], self.XD[test_idx]
            self.Y_inv_train, self.Y_inv_test = self.Y_inv[train_idx], self.Y_inv[test_idx]
            self.Y_train, self.Y_test = self.Y[train_idx], self.Y[test_idx]

            # weight for average
            self.weight_list.append(len(test_idx) / self.N)

            '''training DR'''

            print('Estimate direct regression estimator...')
            est_Y_inv_list = self.DR_estimate(self.XD_train, self.Y_inv_train, self.XD_test, self.Y_inv_test, self.Y_train, self.Y_test)

            for i, d in enumerate(self.conter_D_test):
                est_DR = np.mean(est_Y_inv_list[i], axis=0)
                self.est_DR_list[i, :, iter] = est_DR

            '''training IPW'''
            print('Estimate generalized propensity score...')
            est_GPS_list = self.GPS_estimate(self.XD_train, self.XD_test)
            # trim the sample with GPS smaller than 0.01 to 0.01
            for est_GPS in est_GPS_list:
                pd.DataFrame(est_GPS).to_excel(os.path.join(args.results_save_path, 'D_pred.xlsx'))
                est_GPS[est_GPS < 0.01] = 0.01

            print('Estimate optimal bandwidth...')
            opt_bandwidth_list = self.Bandwidth_estimate(est_GPS_list, est_Y_inv_list, self.XD_test, self.Y_inv_test, kernel_met=self.kernel_met)

            print('Estimate inverse propensity weighting estimator...')
            for i, d in enumerate(self.conter_D_test):
                K_h = opth.kernel(kernel_met=self.kernel_met, h=opt_bandwidth_list[i], a=d, D=np.expand_dims(self.XD_test[:, -1], 1))
                est_IPW = np.mean((K_h / est_GPS_list[i]) * self.Y_inv_test, axis=0)

                self.est_IPW_list[i, :, iter] = est_IPW

            '''training DML'''
            print('Estimate double machine learning estimator...')
            for i, d in enumerate(self.conter_D_test):
                K_h = opth.kernel(kernel_met=self.kernel_met, h=opt_bandwidth_list[i], a=d, D=np.expand_dims(self.XD_test[:, -1], 1))
                est_DML = np.mean(est_Y_inv_list[i] + (K_h / est_GPS_list[i]) * (self.Y_inv_test - est_Y_inv_list[i]), axis=0)
                self.est_DML_list[i, :, iter] = est_DML

            iter += 1

        return np.average(np.stack(self.est_DR_list), axis=2, weights=self.weight_list), \
               np.average(np.stack(self.est_IPW_list), axis=2, weights=self.weight_list), \
               np.average(np.stack(self.est_DML_list), axis=2, weights=self.weight_list)

    def compute_ATE(self, est_list):
        ATE = np.ones([self.num_treatment, self.num_treatment, self.args.obs_dim])
        for i in range(len(est_list)):
            for j in range(len(est_list)):
                if i != j:
                    ATE[i, j, :] = est_list[i] - est_list[j]
        return ATE


if __name__ == '__main__':
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if not os.path.exists(os.path.join(args.results_save_path, 'Simulation', 'Fold='+str(args.K_fold))):
        os.makedirs(os.path.join(args.results_save_path, 'Simulation', 'Fold='+str(args.K_fold)))

    quantiles = np.linspace(0.1, 0.9, args.obs_dim)


    DGP = DataGenerator(args)
    X, D, Y = DGP.generate()
    # DGP.plot(num_sample=5)
    Ground_truth = DGP.groud_truth(quantiles)

    # estimate the distribution of observations (y)
    y_inv_sample_list, y_inv_pdf_list = [], []
    y_inv_list = []

    # estimate the empirical CDF
    for i in range(len(Y)):
        y_inv = est_ECDF(Y[i, :].squeeze(), quantiles)
        y_inv_list.append(y_inv)

    y_inv_lambda = np.stack(y_inv_list)
    df_y_inv_lambda = pd.DataFrame(y_inv_lambda, columns=['q_10', 'q_20', 'q_30', 'q_40', 'q_50', 'q_60', 'q_70', 'q_80', 'q_90'])
    df_y_inv_lambda.to_excel(os.path.join(args.results_save_path, 'y_inv_ECDF.xlsx'))

    for exp_id in range(args.start_exp, args.end_exp):
        print(f'---------Exp_id: {exp_id}----------')
        # df_res = pd.DataFrame(columns=['MCP', 'lasso', 'ridge', 'elastic net', 'NFR', 'rep_NFR'])
        df_res = pd.DataFrame(columns=['q_10', 'q_20', 'q_30', 'q_40', 'q_50', 'q_60', 'q_70', 'q_80', 'q_90'])

        estimator = Estimator(Y_inv=y_inv_lambda, D=D, X=X, Y=Y, quantiles=quantiles, K_fold=args.K_fold, reg_met=args.reg_met)

        DR_res, IPW_res, DML_res = estimator.estimate()
        for i, d in enumerate([-0.05, 0.00, 0.05]):
            df_res.loc[f'GROUND_{d}', :] = Ground_truth[i, :]
            df_res.loc[f'DR_{d}', :] = DR_res.squeeze()[i, :]
            df_res.loc[f'IPW_{d}', :] = IPW_res.squeeze()[i, :]
            df_res.loc[f'DML_{d}', :] = DML_res.squeeze()[i, :]

        save_path = os.path.join(args.results_save_path, 'Simulation', 'Size='+str(args.sample_size))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_res.to_excel(os.path.join(save_path, 'exp_'+str(exp_id)+'.xlsx'))
