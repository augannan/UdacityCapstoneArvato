#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:23:38 2018

@author: bmsbm
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def clean_data_dictionary(data_dictionary, df):
    '''
    Creates a new "missing_or_unknown" column in data dictionary, 
    with the string values converted to integers where appropriate.
    
    INPUT: 
        - data_dictionary (DataFrame) : data dictionary
        - df (DataFrame) : Dataset which dictionary interprets
    
    OUTPUT:
        - clean_data_dictionary (DataFrame) : clean version of data dictionary,
            ready for use

    '''
    missing_values = []
    for i in range(data_dictionary.shape[0]):
        # attribute = data_dictionary["attribute"][i]
        sth = [value for value in data_dictionary["missing_or_unknown"][i].\
                                    replace("[","").replace("]","").split(",")]
        if 'XX' in sth:
            missing_values.append(sth)
        elif 'X' in sth:
            missing_values.append(sth)
        elif sth=='':
            missing_values.append(np.nan)
        else:
            missing_values.append(pd.to_numeric(sth))

    data_dictionary['missing_or_unknown'] = missing_values

    data_dictionary.loc[57, 'missing_or_unknown'] = [-1, 'X']
    data_dictionary.loc[58, 'missing_or_unknown'] = ['XX']
    data_dictionary.loc[59, 'missing_or_unknown'] = [-1, 'XX']
    
    attributes_with_meaning = data_dictionary["attribute"]
    all_attributes = list(df)


    # in dictionary but not in dataset
    meaning_wo_attributes = list(set(attributes_with_meaning) -\
                                 set(all_attributes)) 

    for row in meaning_wo_attributes:
        ind = data_dictionary[data_dictionary.attribute == row].index
        data_dictionary.drop(ind, inplace=True)

    data_dictionary.reset_index(drop=True, inplace=True)
    data_dictionary.reset_index(drop=True, inplace=True)
    
    return data_dictionary

def replace_missing_values(df, data_dictionary):
    '''
    Looks up missing values for each dataframe attribute in data dictionary, 
    and replaces with NaN
    
    INPUT:
        - df (DataFrame): DataFrame with missing values
        - data_dictionary (DataFrame): Data dictionary
    OUTPUT:
        None
    '''
    for i in range(data_dictionary.shape[0]):
        attribute = data_dictionary["attribute"][i]
        df[attribute] = df[attribute].\
                    replace(data_dictionary["missing_or_unknown"][i], np.nan)


def show_dist_missing_values_column(df, n=50):
    '''
    Shows the distribution of proportion of missing values in a dataset - 
    shows the first n columns with the highest proportion of missing values.
    
    INPUT: 
        - df(DataFrame): dataframe which you want to investigate
        - n (int): the number of attributes you want tosee
    
    OUTPUT: 
        - bar chart showing the first n columns with the highest proportion of 
        missing values
    '''
    n_missing_column_data = df.isnull().sum(axis=0).\
                                            sort_values(ascending=False)[:n]
    prop_missing_column_data = n_missing_column_data/df.shape[0]
    fig = plt.figure(figsize=(25,10))
    ax = prop_missing_column_data.plot.bar()
    ax.set_title("Distribution of missing data in columns of dataset")
    ax.set_xlabel("Attributes")
    ax.set_ylabel("Proportion of missing data points")
    plt.show()

def compare_distribution(column_name, df1, df2):
    '''
    Compares the distribution of a particular attribute between two dataframes
    
    INPUTS:
        - column_name (str): name of attribute
        - df1 (DataFrame): the first dataframe
        - df2 (DataFrame): the second dataframe

    OUTPUT:
         - plots of distribution of attribute between both dataframes
    '''
    
    fig = plt.figure(figsize=(10,5))
    fig.suptitle("Comparison of datasets with and without missing values")
    #fig.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    plt.subplot(1,2,1)
    ax1 = sns.countplot(data = df1, x=df1[column_name])
    ax1.set_xlabel("Values")
    ax1.set_ylabel("Counts")
    ax1.set_title("Dataset with no missing values")
    plt.subplot(1,2,2)
    ax2 = sns.countplot(data=df2, x=df2[column_name])
    ax2.set_title("Dataset with some missing values")
    ax2.set_xlabel("Values")
    ax2.set_ylabel("Counts")
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, wspace=0.4)
    plt.show()

def attributes_by_type(df, data_dictionary):
    '''
    Shows features of different types in the dataset
    
    INPUT:
         - df (DataFrame): Dataset we are investigating
         - data_dictionary (DataFrame) : Data Dictionary
    OUTPUT:
        - cat_features, ord_features, num_features, mixed_features, 
            interval_features
    '''
    cat_features = [each for each in data_dictionary[data_dictionary["type"]=="categorical"]["attribute"]
                    if each in list(df)]
    ord_features = [each for each in data_dictionary[data_dictionary["type"]=="ordinal"]["attribute"]
                    if each in list(df)]
    num_features = [each for each in data_dictionary[data_dictionary["type"]=="numeric"]["attribute"]
                    if each in list(df)]
    mixed_features = [each for each in data_dictionary[data_dictionary["type"]=="mixed"]["attribute"]
                      if each in list(df)]
    interval_features = [each for each in data_dictionary[data_dictionary["type"]=="interval"]["attribute"]
                         if each in list(df)]

    return cat_features, ord_features, num_features, mixed_features, interval_features

def scree_plot(pca):
    '''
    plots plot of components of given pca and the cumulated proportion 
    of variance explained
    
    INPUT:
        - pca : pca model
        
    OUTPUT:
        - plot of pca components against cumulative variance explained
    
    '''
    N = len(pca.explained_variance_ratio_)
    x = range(N)
    y1 = np.cumsum(pca.explained_variance_ratio_)


    fig = plt.figure(figsize=(10,5))
    sns.set_style("ticks")
    plt.plot(x,y1, '-', color='b', label='Explalined Variance Ratio')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.title("Cumulative variance accounted for by principal components")
    sns.despine()
    plt.legend(loc="lower right")
    plt.show()
    return y1

def pca_variance_explained(pca):
    '''
    plots plot of components of given pca and the proportion of variance 
    explained
    
    INPUT:
        - pca : pca model
        
    OUTPUT:
        - plot of pca components against variance explained
    
    '''
    fig = plt.figure(figsize=(10,5))
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.xlabel("Component")
    plt.ylabel("Proportion of variance explained")
    plt.title("Variance accounted for by each principal component")
    sns.despine()
    plt.show()

def pca_results(pca, i, columns, num=3):
    '''
    Map weights for the first principal component to corresponding feature 
    names and then print the linked values, sorted by weight.
    
    INPUTS:
        - pca (pca model): fitted pca model
        - i (int): the integer position of the component
        - num(int): number of (positively and negatively correlated) components
        - columns (list): list of column names
        
    OUTPUT: dataframe with components and their weights, sorted by weight.
    
    '''
    df = pd.DataFrame(pca.components_,columns=columns)
    component = df.iloc[i].sort_values()
    neg_assoc = component.head(num)
    pos_assoc = component.tail(num)
    return pd.concat([neg_assoc, pos_assoc], axis=0)

def clean_data(df, data_dictionary):
    """
        Perform feature trimming, re-encoding, and engineering for demographics
        data
        
        INPUT: Demographics DataFrame
        
        OUTPUT: Trimmed and cleaned demographics DataFrame
        """
    
    try:
        df.drop(['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'],
                axis=1,
                inplace=True)
    except:
        pass

    for i in range(data_dictionary.shape[0]):
        attribute = data_dictionary["attribute"][i]
        df[attribute] = df[attribute].replace(data_dictionary["missing_or_unknown"][i], np.nan)

    print("Drop columns with more than 20% missing values")
    columns_to_drop = ['AGER_TYP',
                       'ALTER_HH',
                       'ALTER_KIND1',
                       'ALTER_KIND2',
                       'ALTER_KIND3',
                       'ALTER_KIND4',
                       'ALTERSKATEGORIE_FEIN',
                       'D19_BANKEN_ONLINE_QUOTE_12',
                       'D19_GESAMT_ONLINE_QUOTE_12',
                       'D19_KONSUMTYP',
                       'D19_LETZTER_KAUF_BRANCHE',
                       'D19_LOTTO',
                       'D19_SOZIALES',
                       'D19_TELKO_ONLINE_QUOTE_12',
                       'D19_VERSAND_ONLINE_QUOTE_12',
                       'D19_VERSI_ONLINE_QUOTE_12',
                       'EXTSEL992', 'GEBURTSJAHR',
                       'KBA05_BAUMAX',
                       'KK_KUNDENTYP',
                       'TITEL_KZ']
    df.drop(columns_to_drop, axis=1, inplace=True)

    print("Drop rows with more than 25 missing values")
    row_error_threshold = 25
    df.dropna(thresh=row_error_threshold, inplace=True, axis=0)
    
    print("Re-encode 'OST_WEST_KZ'")
    df.replace({'OST_WEST_KZ':{'W': 1, 'O': 0}}, inplace=True)
    
    cat_columns_to_drop = ['CAMEO_DEU_2015',
                           'LP_STATUS_FEIN',
                           'GEBAEUDETYP',
                           'LP_FAMILIE_FEIN']
        

    cat_features, ord_features, num_features, mixed_features, interval_features = attributes_by_type(df, data_dictionary)

    print("Imputing missing values for features")
    cat_pipeline = Pipeline([('categorical_impute',
                             SimpleImputer(missing_values=np.nan,
                                           strategy='most_frequent'))])
    num_pipeline = Pipeline([('num_impute',
                             SimpleImputer(missing_values=np.nan,
                                           strategy='median'))])


    transformers = [('cat', cat_pipeline, cat_features),
                   ('num', num_pipeline, num_features),
                   ('mixed', cat_pipeline, mixed_features),
                   ('ord', num_pipeline, ord_features),
                   ('interval', cat_pipeline, interval_features),
                   ]

    ct = ColumnTransformer(transformers=transformers)

    df = ct.fit_transform(df)

    column_names = cat_features + num_features + mixed_features + ord_features + interval_features
    df = pd.DataFrame(df, columns = column_names)

    print("One-hot encoding of categorical variables")
    df = pd.get_dummies(df, columns=['NATIONALITAET_KZ',
                                    'CJT_GESAMTTYP',
                                    'FINANZTYP',
                                    'GFK_URLAUBERTYP',
                                    'LP_FAMILIE_GROB',
                                    'LP_STATUS_GROB',
                                    'SHOPPER_TYP',
                                    'ZABEOTYP',
                                    'CAMEO_DEUG_2015',
                                    'KBA05_MAXHERST'],
                       sparse=True)
    print("Re-encoding categorical features done.")

    print("Engineering mixed-type features")
       
    df['PJ_decade'] = df['PRAEGENDE_JUGENDJAHRE']
    df['PJ_movement'] = df['PRAEGENDE_JUGENDJAHRE']

    decade_values = {2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5,
       13:5, 14:6, 15:6}
    df['PJ_decade'].replace(decade_values, inplace=True)

    movement_values = {3:1, 5:1, 8:1 ,10:1, 12:1, 14:1,  4:2, 6:2, 7:2, 9:2,
    11:2, 13:2, 15:2}
    df['PJ_movement'].replace(movement_values, inplace=True)

    df['CAMEO_INTL_wealth'] = df['CAMEO_INTL_2015']
    df['CAMEO_INTL_life_stage'] = df['CAMEO_INTL_2015']

    life_topology_values={}
    wealth_values = {}
    try:
        for value in df['CAMEO_INTL_2015'].unique():
            life_topology_values[value]=divmod(int(value),10)[1]
            wealth_values[value] = divmod(int(value),10)[0]
    except ValueError:
        pass
    
    df['CAMEO_INTL_wealth'].replace(wealth_values, inplace=True)
    df['CAMEO_INTL_life_stage'].replace(life_topology_values, inplace=True)

    building_types = {2:1, 3:1, 4:1, 5:2}

    df['PLZ8_BAUMAX_buildings'] = df['PLZ8_BAUMAX']
    df['PLZ8_BAUMAX_buildings'].replace(building_types, inplace=True)

    cat_columns_to_drop.extend(['PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE',
    'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB',
    'CAMEO_DEU_2015', 'CAMEO_INTL_2015'])

    print("Engineering mixed-type features done")

    df = df.drop(cat_columns_to_drop, axis=1)

    # Return the cleaned dataframe.
    return df
