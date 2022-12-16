import os
import pandas as pd
from tqdm import tqdm


def load_results(generate_result_fn, experiments=None, results_file=None, overwrite=False):
    if experiments is None and results_file is None:
        raise ValueError('Need to provide experiments or/and results_file')

    if experiments is None and not os.path.exists(results_file):
        raise ValueError('Results file does not exist')

    if overwrite or (results_file is None) or (results_file is not None and not os.path.exists(results_file)):
        if experiments is None:
            raise ValueError('Need to provide experiments')

        # Generate data
        result_dicts = []

        for exp in tqdm(experiments):
            result_dicts.append(generate_result_fn(exp))

        results = pd.DataFrame(result_dicts)

        if results_file is not None:
            directory, filename = os.path.split(results_file)
            if not os.path.exists(directory):
                os.mkdir(directory)

            pd.to_pickle(results, results_file)

    else:
        if not os.path.exists(results_file):
            raise ValueError('Results file does not exist')

        results = pd.read_pickle(results_file)

    return results
