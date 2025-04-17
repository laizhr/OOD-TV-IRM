from .infer_irmv1 import InferIrmV1
from .erm import ERM 
from .infer_irmv1_multi_class import Infer_Irmv1_Multi_Class
from .eiil import EIIL 
from .lff import LfF 
from .irmv1 import IrmV1
from .inter_imv1_tvl1 import InferIrmV1TVL1
from .irmv1_tvl1 import IrmV1TVL1
from .inter_irmv1_multi_class_tvl1 import Infer_Irmv1_Multi_Class_TVL1
from .irmv1_multi_class_tvl1 import Irmv1_Multi_Class_TVL1
from .irmv1_multi_class import Irmv1_Multi_Class

def algorithm_builder(flags, dp):
    class_name = flags.irm_type
    return {
        'infer_irmv1': InferIrmV1,
        'erm': ERM,
        'infer_irmv1_multi_class': Infer_Irmv1_Multi_Class,
        'eiil': EIIL, 
        'lff': LfF,
        "irmv1": IrmV1, 
        "infer_irmv1_tvl1": InferIrmV1TVL1,
        "irmv1_tvl1": IrmV1TVL1,
        "infer_irmv1_multi_class_tvl1": Infer_Irmv1_Multi_Class_TVL1,
        "irmv1_multi_class_tvl1": Irmv1_Multi_Class_TVL1,
        "irmv1_multi_class": Irmv1_Multi_Class,
    }[class_name](flags, dp)