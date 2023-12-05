from anylearn.config import init_sdk
from anylearn.interfaces.resource import SyncResourceUploader
init_sdk('http://anylearn.nelbds.cn', '', '')

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from anylearn.applications.quickstart import quick_train

train_task, algo, dset, project = quick_train(
    algorithm_cloud_name="CNN_homework",
    algorithm_local_dir='.',
    dataset_id="DSETcbef52c447a582ec90ce16ee9f3e",
    algorithm_entrypoint="python -u main.py",
    algorithm_output="save_dir",
    dataset_hyperparam_name="data_dir",
    algorithm_force_update=True,
    algorithm_hyperparams={},
    quota_group_request={
            'name': 'DL2023',
            'RTX-3090-shared': 1,
            'CPU': 10,
            'Memory': 40,
        },
    _uploader=SyncResourceUploader(),
    image_name="QUICKSTART_PYTORCH2.0.1_CUDA11.7_PYTHON3.11"
)

print(train_task)








