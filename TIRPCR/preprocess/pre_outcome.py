from collections import defaultdict
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
import pickle


def is_valid_outcome_range(dx, code_range):
    return any(dx.startswith(code) for code in code_range)


def pre_user_cohort_outcome(indir, patient_list, codes9, codes0):
    cad_user_cohort_outcome = defaultdict(list)
    inpatient_dir = os.path.join(indir, 'inpatient')
    outpatient_dir = os.path.join(indir, 'outpatient')

    files = [os.path.join(inpatient_dir, file) for file in os.listdir(inpatient_dir)]
    files += [os.path.join(outpatient_dir, file) for file in os.listdir(outpatient_dir)]

    DXVER_dict = {'9': codes9, '0': codes0}

    for file in files:
        inpat = pd.read_csv(file, dtype=str)
        DATE_NAME = next(col for col in inpat.columns if 'DATE' in col)
        inpat = inpat[inpat['ENROLID'].isin(patient_list)]
        inpat = inpat[~inpat[DATE_NAME].isnull()]
        DX_col = [col for col in inpat.columns if 'DX' in col]

        for index, row in tqdm(inpat.iterrows(), total=len(inpat)):
            enrolid = row['ENROLID']
            date = row[DATE_NAME]
            DXVER = row.get('DXVER', '0').split('.')[0] 

            if DXVER in DXVER_dict:
                for dx in row[DX_col].dropna():
                    if is_valid_outcome_range(dx, DXVER_dict[DXVER]):
                        cad_user_cohort_outcome[enrolid].append(date)

    return cad_user_cohort_outcome

