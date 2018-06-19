import dlopt.util as ut
import pandas as pd
import numpy as np
from datetime import datetime

class RubbishDataLoader(ut.DataLoader):
    """ Load the rubbish dataset
    filename    file containing the data
    scaler      sklearn scaler funtion to be used with the data
                e.g. MinMaxScaler
    """
    params = {'filename': None,
              'scaler_class': None}
    scaler = None

    def load(self,
             **kwargs):
        self.params.update(kwargs)
        if self.params['filename'] is None:
            raise Exception("A 'filename' must be provided")
        df = pd.read_csv(self.params['filename'])
        df = df.set_index(pd.to_datetime(df['date'].values))
        df.drop(['date'], axis=1, inplace=True)
        df = df / 100
        if self.params['scaler_class'] is not None:
            scaler_class = ut.load_class_from_str(self.params['scaler_class'])
            self.scaler = scaler_class()
            df = pd.DataFrame(self.scaler.fit_transform(df),
                              columns=df.columns)
        return df

    def inverse_transform(self,
                          df):
        if self.scaler is None:
            return df
        else:
            inversed = pd.DataFrame(self.scaler.inverse_transform(df),
                                    columns=df.columns)
            return inversed
