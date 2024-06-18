import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model
from sklearn.model_selection import KFold

'''This part of model aims to estimate the empirical distribution'''


def est_ECDF(y, quantiles):
    num_obs = len(y)
    cdf = np.linspace(0, 1, num_obs)
    y = np.sort(y)
    quantile_list = []

    for q in quantiles:
        quantile_list.append(y[(np.array(cdf) <= q).sum()])
    return quantile_list


'''This part of models aim to do function-on-scalar regression'''

def mean_square_loss(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    return(1/2 *  np.sum((y_pred - y_true)**2))

class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, Phi, lambda_para, gamma_para=3, loss_function=mean_square_loss, X=None, Y=None, B_init=None):
        self.B = None
        self.loss_function = loss_function
        self.B_init = B_init
        self.Phi = Phi
        self.X = X
        self.X_arg = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X_reg = np.kron(self.X_arg, self.Phi)
        self.Y = Y
        self.Y_reg = np.transpose(Y).flatten('F')
        self.p = X.shape[1]
        self.sample_size = X.shape[0]
        self.D = Phi.shape[0]
        self.K = Phi.shape[1]
        self.gamma_para = gamma_para
        self.lambda_para = lambda_para
        
    def predict(self, X):
        # prediction = np.matmul(X, np.transpose(self.B).flatten('F'))
        prediction = np.matmul(X, self.B)
        return(prediction)

    def model_error(self):
        error = self.loss_function(self.predict(self.X_reg), self.Y_reg)
        return(error)
    
    def l2_regularized_loss(self, B):
        self.B = B
        # convert B to the matrix given in the paper
        B_mat_T = np.transpose(B.reshape(self.p+1, self.K)); B_mat = np.transpose(B_mat_T); reg_sum = 0
        for j in range(1, self.p+1):
            B_matj = B_mat[j, :]
            if np.linalg.norm(B_matj, ord=2) <= self.gamma_para * self.lambda_para:
                reg_j = self.lambda_para * np.linalg.norm(B_matj, ord=2) - np.linalg.norm(B_matj, ord=2) ** 2 / (2 * self.lambda_para)
            else:
                reg_j = 1 / 2 * self.gamma_para * self.lambda_para ** 2
            
            reg_sum += reg_j
        reg_sum = self.sample_size * self.D * reg_sum
        return(self.model_error() + reg_sum)
    
    def fit(self, maxiter=250):
        # Initialize B estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.B_init)==type(None):
            # set B_init = 1 for every feature
            self.B_init = np.array([1]*((self.p + 1) * self.K))
        else: 
            # Use provided initial values
            pass
            
        if self.B!=None and all(self.B_init == self.B):
            print("Model already fit once; continuing fit with more itrations.")
            
        res = minimize(self.l2_regularized_loss, self.B_init,
                       method='BFGS', options={'maxiter': 500})
        self.B = res.x
        self.B_init = self.B

    def forecast(self, est_XD):
        est_XD = np.hstack([np.ones((est_XD.shape[0], 1)), est_XD])
        self.B = self.B.reshape(est_XD.shape[1], -1)
        prediction = np.matmul(np.matmul(est_XD, self.B), np.transpose(self.Phi))
        return prediction


class CustomCrossValidator:
    """
    Cross validates arbitrary model using MAPE criterion on
    list of lambdas.
    """
    def __init__(self, X, Y, Phi, ModelClass, loss_function=mean_square_loss):
        
        self.X = X
        self.Y = Y
        self.Phi = Phi
        self.ModelClass = ModelClass
        self.loss_function = loss_function
        self.p = X.shape[1]
        self.D = Phi.shape[0]
        self.K = Phi.shape[1]
    
    def cross_validate(self, lambdas, valid_loss_met, num_folds=3):
        """
        lambdas: set of regularization parameters to try
        num_folds: number of folds to cross-validate against
        """
        
        self.lambdas = lambdas
        self.cv_scores = []
        X = self.X
        Y = self.Y
        Phi = self.Phi
        self.valid_loss_met = valid_loss_met
        
        # B values are not likely to differ dramatically
        # between differnt folds. Keeping track of the estimated
        # B coefficients and passing them as starting values
        # to the .fit() operator on our model class can significantly
        # lower the time it takes for the minimize() function to run
        B_init = None
        
        for lambda_para in self.lambdas:
            #print("Lambda: {}".format(lambda_para))
            
            # Split data into training/holdout sets
            kf = KFold(n_splits=num_folds, shuffle=True)
            kf.get_n_splits(X)
            
            # Keep track of the error for each holdout fold
            k_fold_scores = []
            
            # Iterate over folds, using k-1 folds for training
            # and the k-th fold for validation
            f = 1
            for train_index, test_index in kf.split(X):
                # Training data
                CV_X = X[train_index,:]
                CV_Y = Y[train_index,:]
                
                # Holdout data
                holdout_X = X[test_index,:]
                holdout_Y = Y[test_index,:]
                
                # Fit model to training sample
                lambda_fold_model = self.ModelClass(Phi, lambda_para, gamma_para=3, 
                                                    loss_function=self.loss_function,
                                                    X=CV_X, Y=CV_Y, B_init=B_init)
                lambda_fold_model.fit()
                
                # Extract B values to pass as B_init to speed up estimation of the next fold
                B_init = lambda_fold_model.B
                
                # Calculate holdout error
                holdout_X_aug = np.hstack((np.ones((holdout_X.shape[0], 1)), holdout_X))
                holdout_X_reg = np.kron(holdout_X_aug, Phi); holdout_Y_reg = np.transpose(holdout_Y).flatten('F')
                fold_preds = lambda_fold_model.predict(holdout_X_reg)
                # use mean_square_loss or simply mean square error loss?
                if valid_loss_met == 'mean_square_loss':
                    fold_mape = mean_square_loss(fold_preds, holdout_Y_reg)
                elif valid_loss_met == 'MCP_loss':
                    B_mat_T = np.transpose(lambda_fold_model.B.shape(self.p+1, self.K)); B_mat = np.transpose(B_mat_T);
                    norm_B = np.sqrt(np.sum(B_mat ** 2, axis=1)); threshould = self.gamma_para * self.lambda_para
                    reg_error_sum = np.sum((self.lambda_para*norm_B-1/(2* self.gamma_para)*norm_B**2)*(norm_B<=threshould)+ 1/2*self.gamma_para * self.lambda_para**2*(norm_B>threshould))

                    # reg_error_sum = 0
                    # for j in range(1, self.p+1):
                    #     B_matj = B_mat[j, :]
                    #     if np.linalg.norm(B_matj, ord=2) <= self.gamma_para * self.lambda_para:
                    #         reg_j = self.lambda_para * np.linalg.norm(B_matj, ord=2) - np.linalg.norm(B_matj, ord=2) ** 2 / (2 * self.lambda_para)
                    #     else:
                    #         reg_j = 1 / 2 * self.gamma_para * self.lambda_para ** 2
                    #
                    #     reg_error_sum += reg_j
                    # reg_error_sum = self.sample_size * self.D * reg_error_sum
                    fold_mape = mean_square_loss(fold_preds, holdout_Y_reg) + reg_error_sum
                    
                k_fold_scores.append(fold_mape)
                #print("Fold: {}. Error: {}".format(f, fold_mape))
                f += 1
            
            # Error associated with each lambda is the average
            # of the errors across the k folds
            lambda_scores = np.mean(k_fold_scores)
            #print("LAMBDA AVERAGE: {}".format(lambda_scores))
            self.cv_scores.append(lambda_scores)
        
        # Optimal lambda is that which minimizes the cross-validation error
        self.lambda_star_index = np.argmin(self.cv_scores)
        self.lambda_star = self.lambdas[self.lambda_star_index]
        #print("\n\n**OPTIMAL LAMBDA: {}**".format(self.lambda_star))
        

class TraditionLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, train_met, Phi, alpha_para, l1_ratio_para=3, X=None, Y=None, B_init=None):
        self.B = None
        self.B_init = B_init
        self.Phi = Phi
        self.X = X
        self.X_arg = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X_reg = np.kron(self.X_arg, self.Phi)
        self.Y = Y
        self.Y_reg = np.transpose(Y).flatten('F')
        self.p = X.shape[1]
        self.sample_size = X.shape[0]
        self.D = Phi.shape[0]
        self.K = Phi.shape[1]
        self.alpha_para = alpha_para
        self.l1_ratio_para = l1_ratio_para
        self.train_met = train_met

    def predict(self, X):
        # prediction = np.matmul(X, np.transpose(self.B).flatten('F'))
        prediction = np.matmul(X, self.B)
        return(prediction)
    
    def fit(self, maxiter=250):
        # Initialize B estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if self.train_met == 'elastic net':
            clf = linear_model.ElasticNet(fit_intercept=False, l1_ratio=self.l1_ratio_para, alpha=self.alpha_para)
            
        else:
            if self.train_met == 'lasso':
                clf = linear_model.Lasso(alpha=self.alpha_para, fit_intercept=False)
            elif self.train_met == 'ridge':
                clf = linear_model.Ridge(alpha=self.alpha_para, fit_intercept=False)
            # elif self.train_met == 'OLS':
            #     clf = linear_model.LinearRegression(fit_intercept=False)
        
        clf.fit(self.X_reg, self.Y_reg)
        self.B = clf.coef_
        self.B_init = self.B

    def forecast(self, est_XD):
        est_XD = np.hstack([np.ones((est_XD.shape[0], 1)), est_XD])
        self.B = self.B.reshape(est_XD.shape[1], -1)
        prediction = np.matmul(np.matmul(est_XD, self.B), np.transpose(self.Phi))
        return prediction

class TraditionCrossValidator:
    """
    Cross validates arbitrary model using MAPE criterion on
    list of lambdas.
    """
    def __init__(self, X, Y, Phi, ModelClass):
        
        self.X = X
        self.Y = Y
        self.Phi = Phi
        self.ModelClass = ModelClass
        self.p = X.shape[1]
        self.D = Phi.shape[0]
        self.K = Phi.shape[1]
    
    # alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]; l1_ratio=[.01, .1, .5, .9, .99]
    def cross_validate(self, train_met, alphas, l1_ratio, valid_loss_met, num_folds=3):
        """
        lambdas: set of regularization parameters to try
        num_folds: number of folds to cross-validate against
        """
        self.train_met = train_met; self.alphas = alphas; self.l1_ratio = l1_ratio
        self.cv_scores = []
        X = self.X
        Y = self.Y
        Phi = self.Phi
        self.valid_loss_met = valid_loss_met
        
        # B values are not likely to differ dramatically
        # between differnt folds. Keeping track of the estimated
        # B coefficients and passing them as starting values
        # to the .fit() operator on our model class can significantly
        # lower the time it takes for the minimize() function to run
        B_init = None
        if train_met == 'elastic net':
            common_list = [[alphas[i], l1_ratio[j]] for j in range(len(l1_ratio)) for i in range(len(alphas))]
        else:
            if train_met == 'lasso' or train_met == 'ridge':
                common_list = alphas

        self.common_list = common_list
        for paras in self.common_list:
            if train_met == 'elastic net':
                alpha_para = paras[0]; l1_ratio_para = paras[1]
                #print("alpha: {}".format(alpha_para), "alpha: {}".format(l1_ratio_para))
            else:
                if train_met == 'lasso' or train_met == 'ridge':
                    alpha_para = paras; l1_ratio_para = None
                    #print("alpha: {}".format(paras))
            
            # Split data into training/holdout sets
            kf = KFold(n_splits=num_folds, shuffle=True)
            kf.get_n_splits(X)
            
            # Keep track of the error for each holdout fold
            k_fold_scores = []
            
            # Iterate over folds, using k-1 folds for training
            # and the k-th fold for validation
            f = 1
            for train_index, test_index in kf.split(X):
                # Training data
                CV_X = X[train_index,:]; CV_Y = Y[train_index, :]
                
                # Holdout data
                holdout_X = X[test_index,:]; holdout_Y = Y[test_index, :]
                
                # Fit model to training sample
                lambda_fold_model = self.ModelClass(train_met, Phi, alpha_para, l1_ratio_para, X=CV_X, Y=CV_Y, B_init=B_init)
                lambda_fold_model.fit()
                
                # Extract B values to pass as B_init to speed up estimation of the next fold
                B_init = lambda_fold_model.B
                # Calculate holdout error
                holdout_X_aug = np.hstack((np.ones((holdout_X.shape[0], 1)), holdout_X))
                holdout_X_reg = np.kron(holdout_X_aug, Phi); holdout_Y_reg = np.transpose(holdout_Y).flatten('F')
                fold_preds = lambda_fold_model.predict(holdout_X_reg)
                # use mean_square_loss
                if valid_loss_met == 'mean_square_loss':
                    fold_mape = mean_square_loss(fold_preds, holdout_Y_reg)
                
                k_fold_scores.append(fold_mape)
                #print("Fold: {}. Error: {}".format( f, fold_mape))
                f += 1
            # Error associated with each lambda is the average
            # of the errors across the k folds
            paras_scores = np.mean(k_fold_scores)
            #print("PARAS AVERAGE: {}".format(paras_scores))
            self.cv_scores.append(paras_scores)
        # Optimal lambda is that which minimizes the cross-validation error
        self.para_star_index = np.argmin(self.cv_scores)
        self.para_star = self.common_list[self.para_star_index]
        if train_met == 'elastic net':
            self.alpha_star = self.para_star[0]; self.l1_ratio_star = self.para_star[1]
            #print("\n\n**OPTIMAL ALPHA: {}**".format(self.alpha_star))
            #print("\n\n**OPTIMAL L1_RATIO: {}**".format(self.l1_ratio_star))
        else:
            if train_met == 'lasso' or train_met == 'ridge':
                self.alpha_star = self.para_star
                #print("\n\n**OPTIMAL ALPHA: {}**".format(self.alpha_star))


def Cross_validate(XD, Y_inv, Phi, valid_loss_met, lambdas=None, alphas=None, l1_ratio=None, reg_met=None):

    if reg_met == 'MCP':
        validator = CustomCrossValidator(XD, Y_inv, Phi, CustomLinearModel, loss_function=mean_square_loss)
        validator.cross_validate(lambdas, valid_loss_met, num_folds=3)
        return validator.lambda_star

    else:
        validator = TraditionCrossValidator(XD, Y_inv, Phi, TraditionLinearModel)
        if reg_met == 'lasso' or reg_met == 'ridge':
            validator.cross_validate(reg_met, alphas, None, valid_loss_met, num_folds=3)
            return validator.alpha_star

        elif reg_met == 'elastic net':
            validator.cross_validate(reg_met, alphas, l1_ratio, valid_loss_met, num_folds=3)
            return validator.alpha_star, validator.l1_ratio_star


def Regression(XD, XD_test, est_XD_list, Y_inv, Y_inv_test, Phi, lambda_star, alpha_star, l1_ratio_star, reg_met):
    if reg_met == 'MCP':
        model = CustomLinearModel(Phi, lambda_star, gamma_para=3, loss_function=mean_square_loss, X=XD, Y=Y_inv)

    else:
        if reg_met == 'lasso' or reg_met == 'ridge':
            model = TraditionLinearModel(reg_met, Phi, alpha_star, None, X=XD, Y=Y_inv)

        elif reg_met == 'elastic net':
            model = TraditionLinearModel(reg_met, Phi, alpha_star, l1_ratio_star, X=XD, Y=Y_inv)

    model.fit()
    est_Y_inv = model.forecast(XD_test)
    #print(f'Loss(MAE)={np.mean(np.abs(est_Y_inv-Y_inv_test))}')

    est_Y_inv_list = []
    for est_XD in est_XD_list:
        est_Y_inv_list.append(model.forecast(est_XD))
    return est_Y_inv_list



def propensity_score(X_train, D_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    model_params = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(1, 10)],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier()
    clf = RandomizedSearchCV(rf, model_params, n_iter=20, cv=3, random_state=42)
    model = clf.fit(X_train, D_train)
    est_prob = model.predict_proba(X_test)
    return est_prob


