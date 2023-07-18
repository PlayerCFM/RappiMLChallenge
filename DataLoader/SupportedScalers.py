from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SupportedScalers(Enum):
    Standard = StandardScaler
    MinMax = MinMaxScaler