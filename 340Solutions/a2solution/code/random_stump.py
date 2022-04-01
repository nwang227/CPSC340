import numpy as np
import utils
from decision_stump import DecisionStumpInfoGain


class RandomStumpInfoGain(DecisionStumpInfoGain):
    
    def fit(self, X, y):
        
        # Randomly select k features.
        # This can be done by randomly permuting
        # the feature indices and taking the first k
        d = X.shape[1]
        n_features_to_choose = int(np.floor(np.sqrt(d)))
        
        chosen_features = np.random.choice(d, n_features_to_choose, replace=False)
                
        DecisionStumpInfoGain.fit(self, X, y, split_features=chosen_features)
