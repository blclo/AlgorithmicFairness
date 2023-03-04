# -*- coding: utf-8 -*-
import numpy as np
from src.data.dataloader import CatalanJuvenileJustice
import re

sensitive_attributes = [
    'V1_sex',
    'V4_area_origin',
    'V8_age',
    'V9_age_at_program_end',
    'V10_date_of_birth_year',
    'V10_date_of_birth_month',
]

sensitive_dict = {}

attributes = CatalanJuvenileJustice.getColumns()

for s in sensitive_attributes:
    idx = [i for i, item in enumerate(attributes) if re.search(s+'+', item)]
    print(attributes[idx[0]])
    sensitive_dict[s] = idx 


def Independence(y_pred, label, features):
    sa = 1
    for sa in sensitive_attributes:
    result = np.sum([(y_pred[i]==1 and features[i][sensitive_dict[sa]]==1) for i in range(len(y_pred))], axis=0)
    for y in y_pred:
        if y==1:



        
    return 0.5
