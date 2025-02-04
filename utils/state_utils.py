import numpy as np

def get_state(data):
    return np.concatenate([
        data.qpos.flat[2:],
        data.qvel.flat,
        data.cinert.flat,
        data.cvel.flat,
        data.qfrc_actuator.flat,
        data.cfrc_ext.flat,
    ])