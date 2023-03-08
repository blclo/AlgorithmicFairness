# -*- coding: utf-8 -*-
import math
import numpy as np
from torch import IntTensor
from src.data.dataloader import CatalanJuvenileJustice
import re
np.seterr(divide='ignore', invalid='ignore')

class Fairness_criteria:

    def __init__(self, columns):

        self.columns = columns

        sensitive_attributes = [
            'V1_sex',
            'V4_area_origin',
            'V8_age',
            'V9_age_at_program_end',
            'V10_date_of_birth_year'
        ]

        self.independence_dict = {}

        self.sensitive_dict = {}

        for s in sensitive_attributes:
            idx = [i for i, item in enumerate(self.columns) if re.findall(s+'+', item)]
            self.sensitive_dict[s] = idx 

    def Independence(self, y_pred, features):
        for key, value in self.sensitive_dict.items():
            for v in value:
                ''' For Debugging <3 
                print(self.columns[v])
                print('y_pred 0.5 ' , (IntTensor.item(y_pred[0])>=0.5) )
                print('y_pred ' , IntTensor.item(y_pred[0]) )
                print('group 1 ' , IntTensor.item(features[0][v])==1 )
                print('group ' , IntTensor.item(features[0][v]))
                '''
                acceptance = np.sum([(IntTensor.item(y_pred[i])>=0.5) and (IntTensor.item(features[i][v])==1) for i in range(len(y_pred))])
                group = np.sum([IntTensor.item(features[i][v])==1 for i in range(len(y_pred))])
                acceptance_rate = acceptance / group
                if (math.isnan(acceptance_rate)):
                    acceptance_rate = 0
                #print('Acceptance:', acceptance)
                #print('Size of group:', group)
                self.independence_dict[self.columns[v]] = acceptance_rate

        return self.independence_dict