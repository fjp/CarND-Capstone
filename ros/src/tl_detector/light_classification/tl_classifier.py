from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #self.model = load_model('light_classifier_model.h5')
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO Implement classification of green and yellow traffic lights
        # Assume unknown traffic light state because the vehicle will stop if this state occures
        state = TrafficLight.UNKNOWN
        output = image.copy()
        redImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Perform thresholding
        # Filter red color for simulator
        lowerRed = np.array([0,50,50])
        upperRed = np.array([10,255,255])
        redSim = cv2.inRange(redImage, lowerRed , upperRed)

        # Filter red color for real test site
        lowerRed = np.array([170,50,50])
        upperRed = np.array([180,255,255])
        redReal = cv2.inRange(redImage, lowerRed , upperRed)
        
        # Combine filtered images using full weights of 1.0 and no offset (gamma = 0)
        combinedRedImage = cv2.addWeighted(redSim, 1.0, redReal, 1.0, 0.0)

        blurredRedImage = cv2.GaussianBlur(combinedRedImage, (15,15), 0)

        # Find circles in the image that contains only red circles
        redCircles = cv2.HoughCircles(blurredRedImage, cv2.HOUGH_GRADIENT, 0.5, 41, param1=70, param2=30, minRadius=5, maxRadius=150)

        if redCircles is not None:
            state = TrafficLight.RED


        return state
