import numpy as np

class SimpleGaitCycleEstimator:
    def __init__(self, delay):
        """
        A simple gait cycle estimator using ground reaction force (GRF).

        Parameters:
        delay (float): Time delay used to simulate phase offset.
        """
        self.horizon = 3  # Must be greater than 1
        self.last_liftoff_time = 0
        self.last_liftoff_durations_mean = 0
        self.gait_cycle_durations = []
        self.delay = delay

    def __call__(self, time, grf):
        """
        Estimates the gait cycle phase based on time and GRF data.

        Parameters:
        time (float): Current time.
        grf (list or np.ndarray): Ground reaction force values.

        Returns:
        float: Estimated gait cycle phase in [0,1].
        """
        grf = np.asarray(grf)

        # Apply delay to time for phase calculation
        delayed_time = time - self.delay

        # Detect liftoff when GRF crosses a threshold from below
        if len(grf) >= 2 and grf[-1] > 1e1 and grf[-2] < 1e1:
            self.gait_cycle_durations.append(time - self.last_liftoff_time)
            self.last_liftoff_time = time

        # Maintain a moving average of the last few gait cycle durations
        if len(self.gait_cycle_durations) > self.horizon:
            self.gait_cycle_durations.pop(0)

        if self.gait_cycle_durations:
            self.last_liftoff_durations_mean = np.mean(self.gait_cycle_durations)

        # Compute gait cycle phase
        if self.last_liftoff_durations_mean > 0:
            gait_cycle = (delayed_time - self.last_liftoff_time) / self.last_liftoff_durations_mean

            # Adjust if delayed_time is before last liftoff
            if delayed_time < self.last_liftoff_time:
                gait_cycle += 1  # Shift cycle forward
        else:
            gait_cycle = 0

        return np.clip(gait_cycle, 0, 1)
