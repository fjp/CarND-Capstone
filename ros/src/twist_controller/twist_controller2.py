from pid import PID
import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


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
        self.max_braking_force =(0.5 * self.kerb_mass* self.main_wheel_radius)

        # Intiate LowPass filters for Throttle, Steering and brake
	# Create longitudinal pid controller
        self.throttle_pid = PID(20, 0.01, 10.0)
	
	# Create provided yaw controller object for lateral control
	self.min_speed = 0
        self.yaw = YawController(self.main_wheel_base,self.main_steer_ratio, self.min_speed, self.main_lat_accel, self.main_max_steer_angle)
        self.steerLowPass= LowPassFilter(0.2,0.01)
	self.Call_number = 0

    def control(self, proposed_linear_vel, proposed_angular_vel, current_linear_vel, sampletime, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # Initialize throttle,  brake and steer values
        throttle = 0.0
        brake = 0.0
        steer = 0.0


	throttleError = proposed_linear_vel - current_linear_vel
	throttle = max(min(self.throttle_pid.step(throttleError, sampletime),1.0),0.0)
	#rospy.loginfo('throttle: %s', throttle)	
        
        if (throttle > 0.0):
            brake = 0.0
        elif (throttle < 0.0):
            brake = self.max_braking_force * min(throttle , self.main_decel_limit)
            throttle = 0
        
	steer = self.yaw.get_steering(proposed_linear_vel, proposed_angular_vel, current_linear_vel)
        #steer = self.steerLowPass.filt(steer_temp) 
        rospy.loginfo('Steering Value: %s ', steer)
        
	return throttle, brake, steer

    def reset(self):
	self.throttle_pid.reset()
