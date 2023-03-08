# -*- coding: utf-8 -*-
import numpy as np
from torch import IntTensor
from src.data.dataloader import CatalanJuvenileJustice
import re

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

    def Independence(self, y_pred, labels, features):
        for key, value in self.sensitive_dict.items():
            for v in value:
                #print(self.columns[v])
                acceptance = np.sum([(IntTensor.item(labels[i])==1) and (IntTensor.item(features[i][v])==1) for i in range(len(y_pred))])
                group = np.sum([IntTensor.item(features[i][v])==1 for i in range(len(y_pred))])

                #positive = np.sum([IntTensor.item(self.y_pred[i])==1 for i in range(len(self.y_pred))])
                #print('Acceptance:', acceptance)
                #print('Size of group:', group)
                self.independence_dict[self.columns[v]] = acceptance / group

                #self.independence_dict[self.columns[v]] = np.sum([(IntTensor.item(self.features[i][v]==1)) for i in range(len(self.y_pred))], axis=0) / np.sum([(positive and IntTensor.item(self.features[i][v]==1)) for i in range(len(self.y_pred))], axis=0)          
        return self.independence_dict