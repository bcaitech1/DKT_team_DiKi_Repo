import os, re, glob, random, torch
from pathlib import Path
import numpy as np

def setSeeds(seed = 42):
    """Set seed

    Args:
        seed (int)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def increment_path(path : str, exist_ok : bool = False) -> str:
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    
    Return:
        Path to save model (str)
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def delete_model(model_dir : str) -> None:
    file_list = os.listdir(model_dir)
    for file in file_list:
        if file.startswith('model'):
            os.remove(os.path.join(model_dir, file))
            break