import numpy as np
import pandas as pd
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#df_gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#df_gender_submission.to_csv('/kaggle/working/gender_submission.csv', index=False)
df_gender_submission = pd.read_csv('./data/gender_submission.csv')
df_gender_submission.to_csv('./data/submit.csv', index=False)