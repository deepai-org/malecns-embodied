"""Encode the existing antagonist law for physics-rate force evaluation."""
import numpy as np


def muscle_coefficients(drive):
    activation = np.asarray(drive.activation)
    if activation.ndim!=3 or activation.shape[1:]!=(len(drive.groups),2):
        raise ValueError('activation shape differs')
    if not np.isfinite(activation).all() or np.any((activation<0)|(activation>1)):
        raise ValueError('invalid activation')
    if drive.polarity not in (-1,1):
        raise ValueError('invalid polarity')
    result = np.zeros((len(activation),84,3),dtype='<f4')
    seen = set()
    for i,g in enumerate(drive.groups):
        channel = g['channel']
        if channel in seen or not 0<=channel<84 or not np.isfinite(g['rest']):
            raise ValueError('invalid muscle joint')
        seen.add(channel)
        flexor,extensor = activation[:,i,0],activation[:,i,1]
        stiffness = .03*(flexor+extensor)
        result[:,channel,0] = .03*drive.polarity*(flexor-extensor)+stiffness*g['rest']
        result[:,channel,1] = stiffness
        result[:,channel,2] = .001*(flexor+extensor)
    if not np.isfinite(result).all() or np.any(np.abs(result)>1):
        raise ValueError('coefficient contract exceeded')
    return result
