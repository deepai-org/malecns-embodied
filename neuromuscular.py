"""Anatomically restricted candidate interface, without a learned decoder.

Equal averaging of normalized motor-neuron rates is an explicit recruitment
assumption, NOT fitted physiology. Unresolved muscles receive no neural drive.
The interface sees no body state, desired posture, time, reward or action label.
"""
import numpy as np


class NamedMuscleDrive:
    def __init__(self, report, motor_rows):
        self.motor_rows = np.asarray(motor_rows, dtype=np.int64)
        if len(set(self.motor_rows.tolist())) != len(self.motor_rows):
            raise ValueError('duplicate motor rows')
        index = {int(row): i for i, row in enumerate(self.motor_rows)}
        self.weights = np.zeros((report['actuators'], len(index)), dtype=np.float32)
        self.names = []
        seen = set()
        for muscle in report['muscles']:
            target = muscle['muscle_id']
            if target in seen or not 0 <= target < report['actuators']:
                raise ValueError('invalid actuator identity')
            seen.add(target)
            rows = muscle['candidate_motor_rows']
            if len(rows) != len(set(rows)):
                raise ValueError('duplicate candidate neuron')
            if rows:
                self.weights[target, [index[row] for row in rows]] = 1 / len(rows)
            self.names.append(muscle['muscle'])
        if len(seen) != report['actuators']:
            raise ValueError('incomplete actuator inventory')

    def __call__(self, rates):
        rates = np.asarray(rates, dtype=np.float32)
        if rates.shape != (len(self.motor_rows),):
            raise ValueError('expected one rate per annotated motor neuron')
        if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
            raise ValueError('motor rates must be finite normalized values')
        return self.weights @ rates
