"""Pooled point-demand estimator plus history-free cold-start estimator."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from .energy_features import matrix,categorical_indices

PARAMETERS=dict(max_iter=80,learning_rate=.07,max_leaf_nodes=15,min_samples_leaf=200,
                l2_regularization=2.,early_stopping=False,random_state=42)


class EnergyModel:
    def __init__(self,columns,loss):
        self.columns=columns;self.loss=loss;self.model=HistGradientBoostingRegressor(
            loss=loss,categorical_features=categorical_indices(columns),**PARAMETERS)
        self.fit_source_max=None
    def fit(self,x,y,source_max):
        if not np.isfinite(y).all():raise ValueError("Missing/synthetic label requested for fit")
        self.model.fit(x,y);self.fit_source_max=source_max;return self
    def predict(self,frame):
        return np.maximum(0.,self.model.predict(matrix(frame,self.columns)))


def completed_baselines(frame,fallback):
    mapping={"A":"PreviousAvailableSameInterval","B":"PreviousWeekSameLocalInterval","C":"RecentSameSlot28dMedian"}
    return {name:np.where(frame[col].notna(),frame[col].to_numpy(dtype=float),fallback) for name,col in mapping.items()}


def apply_blend(predictions,weights):
    return sum(predictions[name]*weight for name,weight in weights.items())
