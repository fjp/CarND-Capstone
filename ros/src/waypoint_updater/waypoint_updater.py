#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

# New imports
import tf
from scipy.spatial import KDTree
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''
LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

# New defines
RATE = 5
TARGET_SPEED_MPH = 20 # Desired velocity in miles per hour
TARGET_SPEED_MPS = (TARGET_SPEED_MPH * 1609.34) / (60 * 60) # Desired velocity in meters per second
MAX_DECEL = .5
LOGGING = False

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Current pose of the vehicle
        self.pose = None
        self.baseWaypoints = None
        self.waypoints2D = None
        self.waypointTree = None 
        self.stopline_wp_idx = -1
        self.rate = rospy.Rate(RATE)		
		
        # Use queue_size=1 to work only on the latest vehicle pose
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # Subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1);

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Removed rospy.spin() to use a loop function instead
        # This gives better control over the publishing frequency
        self.loop()


    def loop(self):
        while not rospy.is_shutdown():
            if self.pose and self.baseWaypoints and self.waypointTree:
                # Find closest waypoint
                nextWaypointIdx = self.NextWaypoint()
                lane = self.generate_lane(nextWaypointIdx)
                self.publish_waypoints(lane)
                self.rate.sleep()



    def NextWaypoint(self):
        x = self.pose.position.x
        y = self.pose.position.y
        # Query the KD tree to give us the first closest item to the current vehicle (x,y) position
        # Because we only need the index of this coordinate we use [1]
        closestIdx = self.waypointTree.query([x, y], 1)[1]

        # Check if closest is ahead or behind the vehicle 
        closestCoord = self.waypoints2D[closestIdx]
        prevCoord = self.waypoints2D[closestIdx-1]
		
        # Equation for hyperplane through closestCords
        closestVec = np.array(closestCoord)
        prevVec = np.array(prevCoord)
        posVec = np.array([x, y])

        val = np.dot(closestVec - prevVec, posVec - closestVec)

        if val > 0: # check if angle between the two vectors is obtuse or acute
            # if the angle is acute the waypoint is behind the vehicle therefore take the next one
            closestIdx = (closestIdx + 1) % len(self.waypoints2D)

        return closestIdx
		

    def generate_lane(self, closest_idx):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)

        farthest_idx = closest_idx + LOOKAHEAD_WPS
        finalWaypoints = self.baseWaypoints[closest_idx:farthest_idx]

	
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            # set the velocity for lookahead waypoints
            #for i in range(len(finalWaypoints) - 1):
            #    # convert 10 miles per hour to meters per sec
            #    self.set_waypoint_velocity(finalWaypoints, i, TARGET_SPEED_MPS)

            lane.waypoints = finalWaypoints
            if (LOGGING): rospy.loginfo('[waypoint updater] No red light, set velocity to %s m/s (%s miles/h)', TARGET_SPEED_MPS, TARGET_SPEED_MPH)
        else:
            lane.waypoints = self.decelerate_waypoints(finalWaypoints, closest_idx)
            if (LOGGING): rospy.loginfo('[waypoint_updater] Red light at %s and vehicle is at %s', self.stopline_wp_idx, self.NextWaypoint()) 

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        # Create a new list of waypoints and do not mess up the base waypoints which are only received once
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # Two waypoints back from stopping line so that the front of the car stops in front of the stopping line
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            # Get the distance between the current way point i and the stop line idx - 2 
            # The Distance function returns zero if the vehicle is behind the stop line
            dist = self.Distance(waypoints, i, stop_idx)
            # Calculate the velocity depending on the distance to the stop line
            # The larger the distance the larger the velocity
            vel = math.sqrt(2 * MAX_DECEL * dist) # TODO: consider using a linear (constant) deceleration instead of the sqrt function
            # Set the velocity to zero if it is smaller than 1 mps
            if vel < 1.:
                vel = 0.
			
            # To respect the maximum speed limit for each base waypoint
            # we avoid large velocity values which are possible depending on the the vehicle distance to the stopping line
            # As a maximum velocity, keep the one that was given before to the current waypoint
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        #rospy.loginfo('[waypoint updater] last vel %s', temp[-1].twist.twist.linear.x)
	return temp


    def publish_waypoints(self, lane):
        self.final_waypoints_pub.publish(lane)


    def pose_cb(self, msg):
        pose = msg.pose
        self.pose = pose
        x = pose.position.x
        y = pose.position.y
        # Convert Quaternion to Euler
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)

        #rospy.loginfo('[waypoint_updater.py] Vehicle Pose Received - x:%s, y:%s, yaw:%s', x, y, yaw)

    def waypoints_cb(self, msg):
        # Publish to /base_waypoints only once
        wpCount = len(msg.waypoints)
        if self.waypoints2D is None:
            self.baseWaypoints = msg.waypoints
            self.waypoints2D = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in msg.waypoints]
            self.waypointTree = KDTree(self.waypoints2D)
            rospy.loginfo('[waypoint updater] Received %s waypoints', wpCount)


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data
        #if self.stopline_wp_idx != -1:
            #rospy.loginfo('[waypoint updater] Received traffic waypoint stop line index: %s', self.stopline_wp_idx) 

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def Distance(self, waypoints, wp1, wp2):
        # waypoints input is a sliced list of all waypoints ranging from the index that is closest to the vehicle up to 
        # the index that is equal to the closest index plus LOOKAHEAD  
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
