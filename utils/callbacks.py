from fastai.torch_core import *
from fastai.basic_train import Learner, LearnerCallback

class SaveBestModel(LearnerCallback):
    def __init__(self, learn:Learner, ckpt_name:str='best_model'):
        super().__init__(learn)
        self.ckpt_name = ckpt_name
        self.learn = learn
        self.best_loss = None
        self.best_metric = None

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        if self.best_metric is None or last_metrics[-1] > self.best_metric:
            self.best_metric = last_metrics[-1]
            self.learn.save(f'{self.ckpt_name}')
        if self.best_loss is None or smooth_loss < self.best_loss:
            self.best_loss = smooth_loss
            self.learn.save(f'best_loss')
