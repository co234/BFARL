import logging
import os.path
import pandas as pd
import numpy as np
import math
from random import seed, shuffle
from scipy.stats import multivariate_normal 
from scipy.stats import bernoulli as ber 

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)
LOG = logging.getLogger('main')


# The data proessing part is adapted from the AIF360: https://github.com/Trusted-AI/AIF360/tree/main/aif360/datasets


class AdultProcess():
    def __init__(self, label_name='income-per-year',
                 protected_attribute_names = ['sex'],
                 privileged_classes=[['Male']],
                 categorical_features=['workclass', 'education',
                     'marital-status', 'occupation', 'relationship',
                     'native-country'],
                 features_to_keep=[], features_to_drop=['fnlwgt'],
                 na_values=['?'],
                 metadata=None):

        self.label_name = label_name
        self.protected_attribute_names = protected_attribute_names
        self.privileged_classes = privileged_classes
        self.categorical_features = categorical_features
        self.features_to_keep = features_to_keep
        self.features_to_drop = features_to_drop
        self.na_values = na_values
        self.metadata = metadata
        

    def process(self,dir):
        
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
    
        df = pd.read_csv(dir, header=None, names=column_names,
                skipinitialspace=True, na_values=self.na_values,sep=',\s*',engine = 'python',skiprows = 1)

        df.replace(['<=50K.', '>50K.'],['<=50K', '>50K'], inplace = True)

        df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
                  'Married-spouse-absent', 'Never-married','Separated','Widowed'],
                 ['not married','married','married','married','not married',
                  'not married','not married'], 
                 inplace = True)

        df.replace(['Husband','Not-in-family','Other-relative',
                        'Own-child','Unmarried','Wife'],
                        ['not single','single','other','single','single','not single'],
                        inplace=True)

        df.replace(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
                        ['private', 'no income', 'self-emp', 'gov','gov','gov',
                        'no income','no income'], inplace = True)

        df.replace(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                        ['white','blue','blue','blue','white','white',
                        'blue','blue','white','blue','blue','blue','blue','blue'],
                        inplace=True)
        df['native-country'] = df['native-country'].map(lambda x:1 if x == 'United-States' else 0)

        cols_norm = ['age', 'education-num', 'capital-gain','capital-loss','hours-per-week']
        df[cols_norm] = df[cols_norm].apply(lambda x: (x-np.mean(x))/np.std(x))
        

        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
                | set(self.categorical_features) | set([self.label_name]) | set(cols_norm))
    
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
    #     categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
    # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        
        for attr, vals in zip(self.protected_attribute_names, self.privileged_classes):
            privileged_values = [0.]
            unprivileged_values = [1.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

        # s = df['sex'].idx

        y = df[self.label_name].map({'<=50K': 0, '>50K': 1}).values
       
        
        df.drop([self.label_name],axis=1,inplace=True)
       
        s = df.columns.get_loc("sex")
        x = np.asarray(df)
     
        # x = torch.from_numpy(x).type(torch.FloatTensor)

        return x,y,s

class ViloentProcess():
    def __init__(self, label_name='event', favorable_classes=[0],
                protected_attribute_names=['sex','race'],
                privileged_classes=[['Female'],['Caucasian']],
                instance_weights_name=None,
                categorical_features=['c_charge_degree',
                    'age_cat'],
                features_to_keep=['sex', 'age_cat', 'race',
                    'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                    'priors_count', 'c_charge_degree', 
                    'event','decile_score'],
                features_to_drop=[], na_values=[]):

                self.label_name = label_name
                self.protected_attribute_names = protected_attribute_names
                self.privileged_classes = privileged_classes
                self.categorical_features = categorical_features
                self.features_to_keep = features_to_keep
                self.features_to_drop = features_to_drop
                self.na_values = na_values
                self.instance_weights_name = instance_weights_name
            

        
    def process(self,dir):
        df = pd.read_csv(dir,index_col='id',na_values=self.na_values)

        df[(df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != 'O')
            & (df.score_text != 'N/A')]

        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
            | set(self.categorical_features) | set([self.label_name]))
        if self.instance_weights_name:
            keep |= set([self.instance_weights_name])
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(self.categorical_features) - set(self.features_to_drop), key=df.columns.get_loc)
        
        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
        # 5. Create a one-hot encoding of the categorical variables.
        # df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        for col in categorical_features:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
        
        for attr, vals in zip(self.protected_attribute_names, self.privileged_classes):
            privileged_values = [0.]
            unprivileged_values = [1.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

        y = df[self.label_name].values
    
        df.drop([self.label_name],axis=1,inplace=True)
    
        s = df.columns.get_loc(self.protected_attribute_names[1])
        x = np.array(df)
    
        

        return x,y,s

class CompasProcess():
    def __init__(self, label_name='two_year_recid', favorable_classes=[0],
                protected_attribute_names=['sex','race'],
                privileged_classes=[['Female']],
                instance_weights_name=None,
                categorical_features=['c_charge_degree',
                    'age_cat'],
                # features_to_keep=['sex', 'age_cat', 'race',
                #     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                #     'priors_count', 'c_charge_degree','c_charge_desc',
                #     'two_year_recid','decile_score'],
                features_to_keep=['sex', 'age_cat', 'race',
                    'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                    'priors_count', 'c_charge_degree',
                    'two_year_recid'],
                features_to_drop=[], na_values=[]):

                self.label_name = label_name
                self.protected_attribute_names = protected_attribute_names
                self.privileged_classes = privileged_classes
                self.categorical_features = categorical_features
                self.features_to_keep = features_to_keep
                self.features_to_drop = features_to_drop
                self.na_values = na_values
                self.instance_weights_name = instance_weights_name
            

        
    def process(self,dir):
        df = pd.read_csv(dir,index_col='id',na_values=self.na_values)
        df['race'] = df['race'].map(lambda x:1 if x =="African-American"  else 0)
    
    
        df[(df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != 'O')
            & (df.score_text != 'N/A')]

        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
            | set(self.categorical_features) | set([self.label_name]))
        if self.instance_weights_name:
            keep |= set([self.instance_weights_name])
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(self.categorical_features) - set(self.features_to_drop), key=df.columns.get_loc)
        
        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        # for col in categorical_features:
        #     df[col] = df[col].astype('category')
        #     df[col] = df[col].cat.codes
        
        
        for attr, vals in zip(self.protected_attribute_names, self.privileged_classes):
            privileged_values = [0.]
            unprivileged_values = [1.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

        y = df[self.label_name].values

    
        df.drop([self.label_name],axis=1,inplace=True)
    
        s = df.columns.get_loc(self.protected_attribute_names[1])
        x = np.array(df)
    
        

        return x,y,s
            



class DrugProcess():
    def __init__(self, label_name='heroin',
                protected_attribute_names=['gender'],
                privileged_classes=[['Female']],
                instance_weights_name=None,
                categorical_features=['education','country','ethnicity'],
                features_to_keep=['age', 'gender', 'education',
                    'country', 'ethnicity', 'nscore',
                    'escore', 'oscore', 'ascore',
                    'cscore','impulsive','ss','heroin'],
                features_to_drop=[], na_values=[]):

                self.label_name = label_name
                self.protected_attribute_names = protected_attribute_names
                self.privileged_classes = privileged_classes
                self.categorical_features = categorical_features
                self.features_to_keep = features_to_keep
                self.features_to_drop = features_to_drop
                self.na_values = na_values
                self.instance_weights_name = instance_weights_name

    def process(self,dir):
        column_names = ['id', 'age', 'gender',
            'education', 'country', 'ethnicity', 'nscore','escore', 'oscore',
            'ascore', 'cscore', 'impulsive', 'ss','alcohol', 'amphet', 'amyl',
            'benzos', 'caff', 'cannabis',
            'choc', 'coke','crack', 'ecstasy', 'heroin','ketamine','legalh','lsd','meth',
            'mushrooms', 'nicotine', 'semer', 'vsa']

        df = pd.read_csv(dir, sep=',', header=None, names=column_names,
                             na_values=self.na_values)
        

        df['heroin'] = df['heroin'].map(lambda x:0 if x =="CL0"  else 1)
        df['gender'] = df['gender'].map(lambda x:1 if x >0 else 0)
     
        
        
        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
            | set(self.categorical_features) | set([self.label_name]))
        if self.instance_weights_name:
            keep |= set([self.instance_weights_name])
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(self.categorical_features) - set(self.features_to_drop), key=df.columns.get_loc)
        
        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
        # 5. Create a one-hot encoding of the categorical variables.
        # df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        for col in categorical_features:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
        
        
        # for attr, vals in zip(self.protected_attribute_names, self.privileged_classes):
        #     privileged_values = [1.]
        #     unprivileged_values = [0.]
        #     if callable(vals):
        #         df[attr] = df[attr].apply(vals)
        #     elif np.issubdtype(df[attr].dtype, np.number):
        #         # this attribute is numeric; no remapping needed
        #         privileged_values = vals
        #         unprivileged_values = list(set(df[attr]).difference(vals))
        #     else:
        #         # find all instances which match any of the attribute values
        #         priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
        #         df.loc[priv, attr] = privileged_values[0]
        #         df.loc[~priv, attr] = unprivileged_values[0]

        y = df[self.label_name].values
    
    
        df.drop([self.label_name],axis=1,inplace=True)
       
       
        s = df.columns.get_loc(self.protected_attribute_names[0])
        x = np.array(df)

        return x,y,s


class GermanProcess():
    def __init__(self, label_name='credit', favorable_classes=[1],
                 protected_attribute_names=['sex'],
                 privileged_classes=[['male']],
                 instance_weights_name=None,
                 categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],
                 features_to_keep=[], 
                 features_to_drop=['personal_status'],
                 na_values=[]):

                self.label_name = label_name
                self.protected_attribute_names = protected_attribute_names
                self.privileged_classes = privileged_classes
                self.categorical_features = categorical_features
                self.features_to_keep = features_to_keep
                self.features_to_drop = features_to_drop
                self.na_values = na_values
                self.instance_weights_name = instance_weights_name


    def process(self,dir):
        column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']

        df = pd.read_csv(dir, sep=' ', header=None, names=column_names,
                             na_values=self.na_values)


        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
        df['sex'] = df['personal_status'].replace(status_map)
        
        label_maps={1.0: 1, 2.0: 0}
        gender_map = {0.0: 'Male', 1.0: 'Female'}
        age_map = {1.0: 'Old', 0.0: 'Young'}
       
        df['sex'] = df['sex'].replace(gender_map)
        df[self.label_name] = df[self.label_name].replace(label_maps)
        # df['age'] = df['age'].map(lambda x:1 if x > 25 else 0)
        
        

        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
            | set(self.categorical_features) | set([self.label_name]))
        if self.instance_weights_name:
            keep |= set([self.instance_weights_name])
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(self.categorical_features) - set(self.features_to_drop), key=df.columns.get_loc)
        
        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
        # 5. Create a one-hot encoding of the categorical variables.
        # df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        for col in categorical_features:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
       
        for attr, vals in zip(self.protected_attribute_names, self.privileged_classes):
            privileged_values = [0.]
            unprivileged_values = [1.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

        y = df[self.label_name].values
    
        df.drop([self.label_name],axis=1,inplace=True)
       
        s = df.columns.get_loc(self.protected_attribute_names[0])
        x = np.array(df)
       

        return x,y,s



class LoanProcess():
    def __init__(self, label_name='Loan_Status',
                 protected_attribute_names = ['Gender'],
                 privileged_classes=[['Male']],
                 categorical_features=['Education', 'Self_Employed',
                     'Married', 'Property_Area'],
                 features_to_keep=[], features_to_drop=['Loan_ID'],
                 na_values=['NaN'],
                 metadata=[]):

        self.label_name = label_name
        self.protected_attribute_names = protected_attribute_names
        self.privileged_classes = privileged_classes
        self.categorical_features = categorical_features
        self.features_to_keep = features_to_keep
        self.features_to_drop = features_to_drop
        self.na_values = na_values
        self.metadata = metadata
        
    

    def process(self,dir):
        df = pd.read_csv(dir)

        cols_norm = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
        df[cols_norm] = df[cols_norm].apply(lambda x: (x-np.mean(x))/np.std(x))
        

        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
                | set(self.categorical_features) | set([self.label_name]) | set(cols_norm))
    
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
    #     categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
    # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        
        for attr, vals in zip(self.protected_attribute_names, self.privileged_classes):
            privileged_values = [0.]
            unprivileged_values = [1.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

        # s = df['sex'].idx

        y = df[self.label_name].map({'Y': 1, 'N': 0}).values
       
        
        df.drop([self.label_name],axis=1,inplace=True)
       
        s = df.columns.get_loc(self.protected_attribute_names[0])
        x = np.array(df)

        return x,y,s

def gen_gaussian(mean_in, cov_in, class_label,n_samples):
    nv = multivariate_normal(mean = mean_in, cov = cov_in)
    X = nv.rvs(n_samples)
    y = np.ones(n_samples, dtype=float) * class_label
    return nv,X,y



        
class ArrhythmiaProcess():
    def __init__(self, label_name='diagnosis',
                instance_weights_name=None,
                categorical_features=[],
                protected_attribute_names = ['sex'],
                features_to_keep=[],
                features_to_drop=['J'], na_values=['?']):

                self.label_name = label_name
                self.categorical_features = categorical_features
                self.protected_attribute_names = protected_attribute_names
                self.features_to_keep = features_to_keep
                self.features_to_drop = features_to_drop
                self.na_values = na_values
                self.instance_weights_name = instance_weights_name

    def process(self,dir):
        df = pd.read_csv(dir, sep=';', na_values=self.na_values)
       

        df['diagnosis'] = df['diagnosis'].map(lambda x:1 if x == 1 else 0)
        
        
        features_to_keep = df.columns.tolist()
        keep = (set(features_to_keep) | set(self.protected_attribute_names)
            | set(self.categorical_features) | set([self.label_name]))
        if self.instance_weights_name:
            keep |= set([self.instance_weights_name])
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(self.categorical_features) - set(self.features_to_drop), key=df.columns.get_loc)
        
        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
    
        y = df[self.label_name].values
    
        df.drop([self.label_name],axis=1,inplace=True)
       
        s = df.columns.get_loc(self.protected_attribute_names[0])
        x = np.array(df)

        return x,y,s
    



class BankProcess():
    def __init__(self, label_name='y', favorable_classes=['yes'],
                protected_attribute_names=['age'],
                instance_weights_name=None,
                categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
                features_to_keep=[],
                features_to_drop=[], na_values=['unknown']):

                self.label_name = label_name
                self.protected_attribute_names = protected_attribute_names
                self.categorical_features = categorical_features
                self.features_to_keep = features_to_keep
                self.features_to_drop = features_to_drop
                self.na_values = na_values
                self.instance_weights_name = instance_weights_name
            

        
    def process(self,dir):
        df = pd.read_csv(dir, sep=';', na_values=self.na_values)
        df['age'] = df['age'].map(lambda x: 1 if x>=25 else 0)
        df['y'] = df['y'].map(lambda x: 1 if x=="yes" else 0)


        features_to_keep = self.features_to_keep or df.columns.tolist()
        keep = (set(self.features_to_keep) | set(self.protected_attribute_names)
            | set(self.categorical_features) | set([self.label_name]))
        if self.instance_weights_name:
            keep |= set([self.instance_weights_name])
        df = df[sorted(keep - set(self.features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(self.categorical_features) - set(self.features_to_drop), key=df.columns.get_loc)
        
        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        # if count > 0:
        #     warning("Missing Data: {} rows removed from {}.".format(count,
        #             type(self).__name__))
        df = dropped
        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')
        # for col in categorical_features:
        #     df[col] = df[col].astype('category')
        #     df[col] = df[col].cat.codes

        y = df[self.label_name].values
    
        df.drop([self.label_name],axis=1,inplace=True)
    
        s = df.columns.get_loc(self.protected_attribute_names[0])
        x = np.array(df)

        # print(x[:,s])

        return x,y,s







  