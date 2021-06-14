import os, random, torch
import numpy as np
def setSeeds(seed=42):
    # 랜덤 시드를 설정하여, 코드를 매번 실행해도 동일한 결과를 얻게 함
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True