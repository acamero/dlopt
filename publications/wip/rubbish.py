import dlopt.util as ut
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class RubbishDataLoader(ut.DataLoader):
    """ Generate a sinusoidal wave
    """
    params = {'filename': None}
    def load(self,
             **kwargs):
        self.params.update(kwargs)
        if self.params['filename'] is None:
            raise Exception("A 'filename' must be provided")
        df = pd.read_csv(self.params['filename'])
        df = df.set_index(pd.to_datetime(df['date'].values))
        df.drop(['date'], axis=1, inplace=True)
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df),
                          columns=df.columns)
        return df
