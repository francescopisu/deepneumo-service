import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from .._typings import ArrayLike, Seed

def three_way_split(data, size, stratify = None, random_state=None):
    """_summary_

    Args:
        data (_type_): _description_
        size (_type_): _description_
        stratify (_type_, optional): _description_. Defaults to None.
        random_state (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    sizes = size.values()
    if not sum(sizes) == 1.0:
        raise ValueError('fractions {0}, {1}, {2} do not add up to 1.0'.format(*sizes))
    
    if isinstance(data, np.ndarray):
        # in this case, columns are range from 0 to N-1 (N is no. of columns)
        data = pd.DataFrame(data)

    if not stratify:
        stratify = data.columns[-1]
    else:
        if not stratify in data.columns:
            raise ValueError(f'{stratify} is not a column in the dataframe')

    # X = df_input # Contains all columns.
    # y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    train_frac, val_frac, test_frac = sizes

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp = train_test_split(data, stratify=data[stratify],
                                        test_size=(1.0 - train_frac),
                                        random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_test_frac = test_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(df_temp, 
                                        stratify=df_temp[stratify],
                                        test_size=relative_test_frac,
                                        random_state=random_state)

    # assert len(data) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test