#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

# Imports
import math
import numpy as np
from scipy.spatial import KDTree

# Defines
LOGGING = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.baseWaypoints = None
        self.waypoints2D = None
        self.waypointTree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, msg):
        # Publish to /base_waypoints only once
        wpCount = len(msg.waypoints)
        if self.baseWaypoints is None:
            self.baseWaypoints = msg.waypoints
            self.waypoints2D = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in msg.waypoints]
            self.waypointTree = KDTree(self.waypoints2D)


    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        # Finds closest visible traffic light, if one exists, and determines its location and color
        # ligth_wp is the index (if no visible traffic sign, then -1)
        # state is the light status: TrafficLight.[RED, YELLOW, GREEN, UNKNOWN]
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        closest_index = -1
        closest_dis = -1

        if self.waypoints is not None:
            waypoints = self.waypoints.waypoints
            for i in range(len(waypoints)):
                #print(pose)
                dis = (waypoints[i].pose.pose.position.x - pose.x) ** 2 + (waypoints[i].pose.pose.position.y - pose.y) ** 2

                if (closest_dis == -1) or (closest_dis > dis):
                    closest_dis = dis
                    closest_index = i
        return closest_index


    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        if (LOGGING): rospy.loginfo('[tl_detector] New image') 
        #cv2.imshow("test",cv_image)
        #k = cv2.waitKey(1)

        # Classify light state
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if (self.baseWaypoints is None):
            return (-1, TrafficLight.UNKNOWN)
        
        light = None
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']


        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.position.x, self.pose.position.y)

        # Find the closest visible traffic light (if one exists)
        diffNumWp = len(self.baseWaypoints)
        for i, light in enumerate(self.lights):  
            # Get stop line waypoint idx
            line = stop_line_positions[i]
            stop_wp_idx = self.get_closest_waypoint(line[0], line[1])
            #stop_wp_idx = 0
            # Find closest stop line waypoint index
            # Number of waypoints between the current (i) stop line idx and 
            # the closest waypoint index to the current vehicle position
            d = stop_wp_idx - car_wp_idx
            if d >= 0 and d < diffNumWp:
                diffNumWp = d
                closest_light = light
                line_wp_idx = stop_wp_idx

        # find the closest visible traffic light (if one exists)
        if closest_light:
            state = self.get_light_state(light)
            if (LOGGING): rospy.loginfo('[tl_detector] Light at %s has state %s', line_wp_idx, state)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
	   using KDTree
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        closestIdx = self.waypointTree.query([x, y], 1)[1]
        return closestIdx

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
