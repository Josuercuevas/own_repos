from configs.conf import (LOGFILES, DEBUG_LEVEL, LOGE, LOGI, RES_PATH)


class BuildTestLDM:
    def __init__(self, modelcheckpoint):
        self.modelcheckpoint = modelcheckpoint

    def check_model(self):
        # load mode
        # dump model information
        # show visualizer
        raise NotImplementedError(f"Module is not ready to be used yet")