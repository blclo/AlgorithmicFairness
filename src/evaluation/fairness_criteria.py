# -*- coding: utf-8 -*-
import numpy as np
from src.data.dataloader import CatalanJuvenileJustice
import re

sensitive_attributes = [
    'V1_sex',
    'V4_area_origin',
    'V8_age',
    'V9_age_at_program_end',
    'V10_date_of_birth_year'
]

independence_dict = {}

sensitive_dict = {}

attributes = ''

def makeSensitive_dict():
    for s in sensitive_attributes:
        idx = [i for i, item in enumerate(attributes) if re.search(s+'+', item)]
        print(attributes[idx[0]])
        sensitive_dict[s] = idx 
    return sensitive_dict

def Independence(y_pred, label, features, columns):
    for key, value in sensitive_dict.items():
        for v in range(len(value)):
            print(attributes[v])
            independence_dict[attributes[v]] = np.sum([(y_pred[i]==1 and features[i][v]==1) for i in range(len(y_pred))], axis=0)
    return independence_dict

def Fairness_criteria(y_pred, labels, features, columns):
    attributes = columns
    
    Independence = Independence(y_pred, labels, features, col)

    return Independence
