from pathlib import Path
def details():
    return {
        'batchSize':20,
        'epochs':30,
        'l_rate':0.0001,
        'inpLang':'en',
        'outLang':'fi',
        'model_folder':'params',
        'basename':'Tformer_model_',
        'preload':'latest',
        'exp_name':'runs/Tformer_model',
        'seqLen':170,
        'tokenizer_file':'tokenizer_{0}.json'
    }

def create_params_path(config,epoch):
    model_basename = config['basename']
    model_folder = config['model_folder']
    filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/filename)

def latest_weights(config):
    model_folder = 'params'
    model_file = f"{config['basename']}*"
    weights_files = list(Path(model_folder).glob(model_file))
    if len(weights_files)==0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

    