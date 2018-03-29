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

# New imports
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

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

        # New local variables
        self.pre_wp_idx = -1  # previous waypoint index
        self.is_classfied_waypoints_by_tl = False
        self.waypoints_to_tl_idx = None # self.waypoints_to_tl_idx[i]=j means that ith waypoint is closet to jth traffic light
        self.car_position = 0  # The index of closest waypoint
        self.is_first_time = True

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg.pose
        self.car_position = self.get_closest_waypoint(self.pose)
        # self.pose.position.[x,y,z]
        # self.pose.orientation.[x,y,z,w] # Quanternion
        # rospy.loginfo('[CSChen] self.pose.position.(x,y,z)=({},{},{})'.format(self.pose.position.x,self.pose.position.y,self.pose.position.z))

    def waypoints_cb(self, msg):
        # Publish to /base_waypoints only once
        wpCount = len(msg.waypoints)
        if self.waypoints is None:
            self.waypoints = msg.waypoints
            rospy.loginfo('[CSChen] Received %s waypoints', wpCount)

    # Find each waypoint's closest traffic lights, (within a distance threshold). Return -1 if no near light
    def classify_waypoints_by_tl(self):
        rtn_waypoints_to_tl_idx = []
        distance_th = 100
        for wp in self.waypoints:
            distlist = [self.distance(wp.pose.pose.position, l.pose.pose.position) for l in self.lights]
            minidx = np.argmin(distlist)
            minvalue = np.min(distlist)
            if minvalue<distance_th:
                rtn_waypoints_to_tl_idx.append(minidx)
            else:
                rtn_waypoints_to_tl_idx.append(-1)
        rospy.loginfo('[CSChen] having {} of -1'.format(rtn_waypoints_to_tl_idx.count(-1)))
        return rtn_waypoints_to_tl_idx

    def traffic_cb(self, msg):
        self.lights = msg.lights  # styx_msgs/TrafficLight[]
        # self.lights[i] is of Type styx_msgs/TrafficLight, has attributes describing bellow:
        # self.lights[i].pose.pose.[position, orientation]
        # self.lights[i].state has velue TrafficLight.[RED, YELLOW, GREEN, UNKNOWN]
        # rospy.loginfo('[CSChen] Received %s traffic lights', len(self.lights))  # 8 traffic lights

        if not self.is_classfied_waypoints_by_tl and not self.waypoints==None:
            self.waypoints_to_tl_idx = self.classify_waypoints_by_tl()
            rospy.loginfo('[CSChen] len(self.waypoints_to_tl_idx)={}. NOTE: This msg should not show up more than twice'.format(len(self.waypoints_to_tl_idx)))
            self.is_classfied_waypoints_by_tl = True

        # rospy.loginfo('[CSChen] Received self.lights[3].pose.pose.position.y={}'.format(self.lights[3].pose.pose.position.y))
        # rospy.loginfo('[CSChen] Received self.lights[3].pose.pose.orientation.w={}'.format(self.lights[3].pose.pose.orientation.w))
        # rospy.loginfo('[CSChen] Received self.lights[3].state={}'.format(self.lights[3].state))



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
        # Debug part
        # if light_wp!=-1:
        #     stop_line_positions = self.config['stop_line_positions']
        #     rospy.loginfo('[CSChen] approaching/leaving {}th traffic light with stopline {}'.format(light_wp,stop_line_positions[light_wp]))

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

    def distance(self, p1, p2):
        # return (p1.x-p2.x)**2 + (p1.y-p2.y)**2
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # Comparing pose.position.[x,y,z] with self.waypoints to find out the closest waypoint
        rtn_idx = -1
        min_dist = float('inf')
        wplen = len(self.waypoints)
        waypoint_search_range = 30

        if self.pre_wp_idx == -1:
            rospy.loginfo('[CSChen] Calculate all waypoints')
            startidx = 0
            endidx = wplen
        else:
            startidx = self.pre_wp_idx-waypoint_search_range
            endidx = self.pre_wp_idx+waypoint_search_range
        for idx in range(startidx,endidx):
            i = idx%wplen
            dist = self.distance(pose.position, self.waypoints[i].pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                rtn_idx = i
        assert(rtn_idx!=-1)
        # rospy.loginfo('[CSChen] rtn_idx={}, min_dist={}'.format(rtn_idx,min_dist))
        self.pre_wp_idx = rtn_idx
        return rtn_idx

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

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        # [[1148.56, 1184.65], [1559.2, 1158.43], [2122.14, 1526.79], [2175.237, 1795.71], [1493.29, 2947.67], [821.96, 2905.8], [161.76, 2303.82], [351.84, 1574.65]]
        stop_line_positions = self.config['stop_line_positions']
        # Debug part
        if self.is_first_time:
            rospy.loginfo('[CSChen] stop_line_positions {}'.format(stop_line_positions))
            self.is_first_time = False

        if(self.pose):
            # because the topic '/current_pose' is updated faster than '/image_color'
            # we can directly use self.car_position which is calculate by self.pose_cb
            car_position = self.car_position

        # Find the closest visible traffic light (if one exists)
        # Should We need to find 'ahead' of traffic light? (not only closest)
        light_idx = self.waypoints_to_tl_idx[car_position]
        if light_idx == -1:
            light = None
        else:
            # rospy.loginfo('[CSChen] approaching/leaving {}th traffic light'.format(light_idx))
            light = self.lights[light_idx]
        if light:
            # TODO: Deep learning traffic sign classifier
            # state = self.get_light_state(light)
            state = light.state  # ground truth label
            # rospy.loginfo('[CSChen] approaching/leaving {}th traffic light with light {}'.format(light_idx,state))
            return light_idx, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
