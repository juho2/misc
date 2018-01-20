#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVC

from time import time

services = [
            'ind_ahor_fin_ult1', #	Saving Account
            'ind_aval_fin_ult1', #	Guarantees
            'ind_cco_fin_ult1', # 	Current Accounts
            'ind_cder_fin_ult1', # 	Derivada Account
            'ind_cno_fin_ult1', # 	Payroll Account
            'ind_ctju_fin_ult1', #	Junior Account
            'ind_ctma_fin_ult1', # 	Más particular Account
            'ind_ctop_fin_ult1', # 	particular Account
            'ind_ctpp_fin_ult1', #	particular Plus Account
            'ind_deco_fin_ult1', # 	Short-term deposits
            'ind_deme_fin_ult1', # 	Medium-term deposits
            'ind_dela_fin_ult1', # 	Long-term deposits
            'ind_ecue_fin_ult1', # 	e-account
            'ind_fond_fin_ult1', #	Funds
            'ind_hip_fin_ult1', # 	Mortgage
            'ind_plan_fin_ult1', #	Pensions
            'ind_pres_fin_ult1', # 	Loans
            'ind_reca_fin_ult1', # 	Taxes
            'ind_tjcr_fin_ult1', # 	Credit Card
            'ind_valo_fin_ult1', # 	Securities
            'ind_viv_fin_ult1', # 	Home Account
            'ind_nomina_ult1', # 	Payroll
            'ind_nom_pens_ult1', # 	Pensions
            'ind_recibo_ult1', # 	Direct Debit
            ]
            
def get_diff(df):
    s = df[services].astype(int)
    s_diff = s.diff()
    s_diff[s_diff<0] = 0
#    for col in ss.columns:
#        if ss[col].any()!=0:
#            print(ss[col], ss_diff[col])
    s_diff.fillna(0, inplace=True)
    return(s_diff)
    
def read_and_pp(readpath, savename, n_rows=None, drop_bad_rows=True):
    if n_rows == None:
        chunk_size=5000000
        reader = pd.read_csv(readpath,
                        dtype={"sexo":str,
                               'age':str,
                               "ind_nuevo":str,
                               'antiguedad':str,
                               'indrel_1mes':str,
                               'conyuemp':str,
                               'tiprel_1mes':str,
                               "ult_fec_cli_1t":str,
                               "indext":str,
                               'canal_entrada':str,
                               'segmento':str,
                               },
                               na_values='         NA',
                                chunksize=chunk_size,
                                header=0,
#                                memory_map=True,
                               )
#        df = pd.concat([preprocess_chunk(chunk) for chunk in reader])
#        with pd.HDFStore('store.h5') as store:
#            store[savename] = df
        num = 0
        for chunk in reader:
#            df = chunk
            df = preprocess_chunk(chunk, drop_bad_rows)
            df.to_hdf('store.h5', savename + str(num))
            num += 1
        return()
    else:
        reader = pd.read_csv(readpath,
                            dtype={"sexo":str,
                                   'age':str,
                                   "ind_nuevo":str,
                                   'antiguedad':str,
                                   'indrel_1mes':str,
                                   'conyuemp':str,
                                   'tiprel_1mes':str,
                                   "ult_fec_cli_1t":str,
                                   "indext":str,
                                   'canal_entrada':str,
                                   'segmento':str,
                                   },
                                   na_values='         NA',
        #                           usecols=range(10), #0-24 personal data, 25-> owned services
        #                            usecols=[0,1]
                                   nrows=n_rows,
        #                            chunksize=chunk_size,
        #                            header=0,
                                    memory_map=True,
                                   )
        df = preprocess_chunk(reader) # if reading all at once

#    print('\nUnique: \n')    
#    for col in df.columns:
#        print(col, df[col].unique())
#    print('\n')
        return(df)

        
def read_and_sample(storepath, varname, samples=None):
    with pd.HDFStore(storepath, mode='r') as store:
        df = store[varname]
#    df = preprocess_full(df)
    if samples==None:
        return(df)
    else:
        # random small sample of unique people for testing
        unique_ids   = pd.Series(df["ncodpers"].unique())
        unique_id    = unique_ids.sample(n=samples)
        df           = df[df.ncodpers.isin(unique_id)]
        return(df)    
    
def fill_most_common(df, col):
    nulls = df[col].isnull().sum()
    if nulls != 0:
        most_common = df[col].value_counts().idxmax()
        df[col].fillna(most_common, inplace=True)
        print('Filled ' + str(nulls) + ' in ' + col + ' by ' + str(most_common))
    return(df)

def drop_NA_by_col(df, col):
    bad_rows = df[col].isnull().sum()
    if bad_rows != 0:
        df.dropna(subset=[col], inplace=True) # assumed bad rows in general
        print('Dropped ' + str(bad_rows) + ' rows by ' + col)
    return(df)

def primary(x):
    if x==1:
        return('Y')
    elif x==99:
        return('ex')
    else:
        return('N')

def c_type(x):
    x=str(x)[0]
    if x=='1':
        return('pri')
    elif x=='2':
        return('coo')
    elif x=='P':
        return('pot')
    elif x=='3':
        return('ex-pri')
    elif x=='4':
        return('ex-coo')
    else:
        return('unknown')

def c_rel_type(x):
    if x in ['A', 'I', 'P', 'R']:
        return(x)
    else:
        return('unknown')

def check_last(x):
    if x.month == 5 and x.year == 2016:
        return(True)
    else:
#        print(pd.DatetimeIndex(x).month)
        return(False)
    
def keep_popular(df, col, th):
#    counts = df[col].value_counts()
#    keep = [x for x in counts.index if counts[x] > th]
    index = df[col].value_counts().index
    keep = [x for x in index[:10]] # 10 most popular 
    print('Kept ' + str(len(keep)) + ' most popular in ' + col)
    df[col] = df[col].apply(lambda x: x if x in keep else 'Rare')
    
    return(df)
        
def preprocess_chunk(df, drop_bad_rows):
    #fecha_dato 	The table is partitioned for this column
    #    dates = ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']
    
    df = df.loc[df.fecha_dato=='2015-05-28' | df.fecha_dato=='2015-06-28' | df.fecha_dato=='2016-05-28' | df.fecha_dato=='2016-06-28']

    df.fecha_dato = pd.to_datetime(df.fecha_dato)
    df["record_month"] = pd.DatetimeIndex(df["fecha_dato"]).month
    df['last_record'] = df.fecha_dato.apply(check_last)    
    df.drop('fecha_dato', axis=1, inplace=True) 

    #ncodpers 	Customer code
    df = df[df.ncodpers!=0] # missing ID => useless
    df = df[df.ncodpers.notnull()]
    
    #ind_empleado 	Employee index: A active, B ex employed, F filial, N not employee, P pasive
    # feature ok as is
    if drop_bad_rows:
        df = drop_NA_by_col(df, 'ind_empleado') # bad rows
        df = df[df.ind_empleado!=0] # more bad rows
    df.loc[df.ind_empleado=='S', 'ind_empleado'] = 'A' # contains some 'S'
    
    ##pais_residencia 	Customer's Country residence
    #df['expat'] = df.pais_residencia.apply(lambda x: 0 if x=='ES' else 1)
    #print('Non-ES residence: ' + str(df.expat.sum()) # stack foreign, not that many
    # redundant with "indresi"? not many non-ES => just drop
    df.drop('pais_residencia', axis=1, inplace=True)
    
    #sexo 	Customer's sex
    df = fill_most_common(df, 'sexo')
    df['woman'] = df.sexo.apply(lambda x: True if x.strip()=='V' else False)
    df.drop('sexo', axis=1, inplace=True)
    
    #age 	Age
    df.age = pd.to_numeric(df.age, errors='coerce')
    df = fill_most_common(df, 'age')
#    plt.figure(0); sns.distplot(df.age.astype(int))
    df['age'] = pd.cut(df.age, [0,18,32,60,150],
                    labels=['child', 'young', 'mid', 'old']).astype(str)
#    df.drop('age', axis=1, inplace=True)
    
    #fecha_alta 	The date in which the customer became as the first holder of a contract in the bank
#    df.fecha_alta = pd.to_datetime(df.fecha_alta)
#    df['join_month'] = pd.DatetimeIndex(df["fecha_alta"]).month
    df.drop('fecha_alta', axis=1, inplace=True)
    
    #ind_nuevo 	New customer Index. 1 if the customer registered in the last 6 months.
    #months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()
    #months_active.max()
    # looks like missing => new
    #df[df.ind_nuevo.isnull()] = 1
    #df.ind_nuevo = df.ind_nuevo.astype(int)
    df.drop('ind_nuevo', axis=1, inplace=True) # redundant with antiguedad
    
    #antiguedad 	Customer seniority (in months)
    df.antiguedad = pd.to_numeric(df.antiguedad, errors='coerce')
    df[(df.antiguedad.isnull()) | (df.antiguedad<0)] = 0 # guessing
    df = fill_most_common(df, 'antiguedad')
#    plt.figure(1); sns.distplot(df.antiguedad.astype(int))
    df.antiguedad = pd.cut(df.antiguedad, [0,7,24,48,300],
                    labels=['new', 'A', 'B', 'C']).astype(str)
#    df.drop('antiguedad', axis=1, inplace=True)
    
    #indrel 	1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
    df['primary'] = df.indrel.apply(primary)
    df.drop('indrel', axis=1, inplace=True)
    
    #ult_fec_cli_1t 	Last date as primary customer (if he isn't at the end of the month)
    # the people with 'indrel'==99, mostly nan but interesting
#    df.ult_fec_cli_1t.describe() # TBD clever feature
    df.drop('ult_fec_cli_1t', axis=1, inplace=True)
    
    #indrel_1mes 	Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
    df['c_type'] = df.indrel_1mes.apply(c_type)
    df.drop('indrel_1mes', axis=1, inplace=True)
    
    #tiprel_1mes 	Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
    df['c_rel_type'] = df.tiprel_1mes.apply(c_rel_type)
    df.drop('tiprel_1mes', axis=1, inplace=True)
    
    #indresi 	Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
    if drop_bad_rows:
        df = drop_NA_by_col(df, 'indresi') # bad rows
    df.indresi = df.indresi.apply(lambda x: True if x=='S' else False)
    
    #indext 	Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
    if drop_bad_rows:
        df = drop_NA_by_col(df, 'indext') # bad rows
    df.indext = df.indext.apply(lambda x: True if x=='S' else False)
    
    #conyuemp 	Spouse index. 1 if the customer is spouse of an employee
    # sparse but keep
    df.conyuemp = df.conyuemp.apply(lambda x: True if x=='S' or x==1 else False)
    
    #indfall 	Deceased index. N/S
    df.indfall = df.indfall.apply(lambda x: True if x=='S' or x==1 else False)
    
    #tipodom 	Addres type. 1, primary address
    df.drop('tipodom', axis=1, inplace=True) # seems useless
    
    #cod_prov 	Province code (customer's address)
    df.drop('cod_prov', axis=1, inplace=True) # redundant with nomprov
    
    #nomprov 	Province name
    df.nomprov.fillna('unknown', inplace=True)
    
    #ind_actividad_cliente 	Activity index (1, active customer; 0, inactive customer)
    df.ind_actividad_cliente = df.ind_actividad_cliente.astype(bool) # feature ok, lots of inactives
    
    #renta 	Gross income of the household
    # many missing, fill by region median. Many 'unknown' seem to be rich, special customers?
    if drop_bad_rows:
        df = df[df.renta != 0] # more bad rows
    incomes = df.loc[df.renta.notnull()][['renta', 'nomprov']].groupby('nomprov').median()
    incomes.sort_values(by=("renta"),inplace=True)
    incomes.reset_index(inplace=True)
    for prov in incomes.nomprov:
        val = incomes.renta[incomes.nomprov==prov].values
        df.loc[(df.renta.isnull()) & (df.nomprov==prov), 'renta'] = val
    df.renta.fillna(incomes.median().values[0], inplace=True) # fill rest by global median
    df.renta = pd.cut(df.renta, [0,60e3,120e3,1e8], labels=['A', 'B', 'C']).astype(str)
#    plt.figure(2); sns.factorplot(x='nomprov', y='renta',data=incomes)
#    plt.xticks(rotation=90)           
    
    return(df)
    
def preprocess_full(df):
    #canal_entrada 	channel used by the customer to join
    #canal entrada, 150 unique, keep >th only
    df = fill_most_common(df, 'canal_entrada')
    df = keep_popular(df, 'canal_entrada', 1e5)
    
    #nomprov, 50 unique keep >th only
    df = keep_popular(df, 'nomprov', 1e5)
    
    #segmento 	segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
    df = fill_most_common(df, 'segmento')
    df["record_month"] = df["record_month"].astype(str)
#    df["join_month"] = df["join_month"].astype(str)
    df = pd.get_dummies(df)
    return(df)
    