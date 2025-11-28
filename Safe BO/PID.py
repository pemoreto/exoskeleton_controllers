class PID:
    def __init__(self, Kp, Ki, Kd, setpoint):
        """
        Initializes a PID controller.

        Parameters:
        Kp (float): Proportional gain
        Ki (float): Integral gain
        Kd (float): Derivative gain
        setpoint (float): Desired target value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.integral = 0
        self.prev_error = 0  # Initialized to 0 instead of None

    def update(self, current_value, dt):
        """
        Updates the PID controller with the current process value.

        Parameters:
        current_value (float): The current measured value
        dt (float): Time step (must be > 0)

        Returns:
        float: The computed control output
        """
        if dt <= 0:
            raise ValueError("dt must be greater than zero to avoid division errors.")

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        self.prev_error = error  # Store error for the next iteration

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        return output
