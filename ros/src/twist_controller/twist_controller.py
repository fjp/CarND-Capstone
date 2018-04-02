from pid import PID
import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
LOGGING = False

class Controller(object):
    def __init__(self, **parameters):

        self.parameter = parameters
        self.main_vehicle_mass = parameters['vehicle_mass']
        self.main_fuel_capacity = parameters['fuel_capacity']
        self.main_brake_deadband = parameters['brake_deadband']
        self.main_decel_limit = parameters['decel_limit']
        self.main_accel_limit = parameters['accel_limit']
        self.main_wheel_radius = parameters['wheel_radius']
        self.main_wheel_base = parameters['wheel_base']
        self.main_steer_ratio = parameters['steer_ratio']
        self.main_lat_accel = parameters['max_lat_accel']
        self.main_max_steer_angle = parameters['max_steer_angle']

        # Caluclate kerb_mass i.e., mass of vehicle with liquids (fuel, etc) minus accessories
        # Assume a fuel level of 100 percent. If you have access to changing Fuel levels, multply value by percentage of fuel present

        self.kerb_mass = self.main_vehicle_mass + (self.main_fuel_capacity * GAS_DENSITY)

        # Calculate Max braking force allowed.
        # assume a 0.5 braking force ratio (including brake pedal to brake force of gear system)
        # mutliply value by wheel radius to account for angular decelartion forces on wheel
        # where angular decelaration is deceleration velocity/wheel radius
        self.max_braking_force =(0.5 * self.kerb_mass* self.main_decel_limit * self.main_wheel_radius)

        # Intiate LowPass filters for Throttle, Steering and brake
        #self.throttle_pid = PID(0.1,0,0.34674,0,1)
        
	# Create longitudinal pid controller
        #self.throttle_pid = PID(0.5,0.00001,0.0,0,1)
        self.throttle_pid = PID(2.0, 0.02, 0.02, parameters['decel_limit'], parameters['accel_limit'])
	
	# Create provided yaw controller object for lateral control
        self.min_speed = 0
        self.yaw = YawController(self.main_wheel_base,self.main_steer_ratio, self.min_speed, self.main_lat_accel, self.main_max_steer_angle)
        self.steerLowPass= LowPassFilter(tau = 3.0 , ts = 1.0)
        self.throttleLowPass = LowPassFilter(tau = 3.0, ts = 1.0)


    def control(self, proposed_linear_vel, proposed_angular_vel, current_linear_vel, sampleTime):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        velError = proposed_linear_vel - current_linear_vel
        throttle = self.throttle_pid.step(velError, sampleTime)
        throttle = self.throttleLowPass.filt(throttle)

        if (throttle > 0.0):
            brake = 0.0
            if (LOGGING): rospy.loginfo('[dbw controller] throttle: %s, brake: %s', throttle, brake)
        else:
            decel = abs(throttle)
            if decel < self.parameter['brake_deadband']:
                decel = 0.0
            brake = decel * (self.parameter['vehicle_mass'] + self.parameter['fuel_capacity'] * GAS_DENSITY) * self.parameter['wheel_radius']

            throttle = 0.0
            if (LOGGING): rospy.loginfo('brake %s, throttle: %s', brake, throttle)

        steer = self.yaw.get_steering(proposed_linear_vel, proposed_angular_vel, current_linear_vel)
        steer = self.steerLowPass.filt(steer) 
        if (LOGGING): rospy.loginfo('[dbw controller] steer: %s ', steer)
        
        return throttle, brake, steer

    def reset(self):
        self.throttle_pid.reset()
