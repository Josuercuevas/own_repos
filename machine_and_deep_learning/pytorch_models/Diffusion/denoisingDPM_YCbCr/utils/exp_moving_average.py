class ExpMovingAverage():
    # reference: https://leimao.github.io/blog/Exponential-Moving-Average/
    def __init__(self, decay):
        self.decay = decay
    
    # this is the key, basically [Old_model*decay + New_model*(1-decay)]
    # decay hear means, how fast you are forgetting the past, a value close
    # to 1 means you are keeping most of the past experiences. This avoids sharp changes
    # and improves model stability
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)