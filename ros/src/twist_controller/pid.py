import rospy
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx
        self.params = []
        self.dp = [1.0,0.000001,0.5]
        self.iteration = 1
        self.i_error = 0
        self.best_error = 100000.0
        self.loop_number = 1
        self.index_param = 2 #which paramter to twiddle
        self.int_val = self.last_error = 0.
        self.prev_param_index = 0
        self.temp_storage2 = []

    def reset(self):
        self.int_val = 0.0

    def update(self, cte):
        #Twiddle Algorithm to settle PID
        if self.iteration == 1:
            self.p_error = cte
	self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error +=cte
        self.params = [self.kp, self.ki, self.kd]
        self.total_error = self.TotalError()
        #rospy.loginfo('Best_error: %s, Total_error: %s', self.best_error, self.total_error)

        if (self.iteration == 1):
            self.best_error = self.total_error
            self.params[self.index_param] += self.dp[self.index_param]
            self.prev_param_index = 1
        else:
            if(self.total_error <= self.best_error): #improvement
                self.loop_number =1 ;
                #rospy.loginfo('loop 3')
                self.best_error = self.total_error
                self.index_param += 1 # move to next paramter twiddling
                if(self.index_param == 3):
                     self.index_param = 0
                self.dp[self.index_param]*=1.1
                #rospy.loginfo('index_param: %s, dp size : %s , params size : %s', self.index_param, len(self.dp), len(self.params))
                self.params[self.index_param] += self.dp[self.index_param]
            else:
                if(self.loop_number !=  1):
                    #rospy.loginfo('loop 5')
                    #self.params[self.index_param] = self.temp_storage
                    #rospy.loginfo('dp %s param %s', self.dp[self.index_param], self.params[self.index_param])
                    if(self.loop_number == 4):
                        #rospy.loginfo('loop 5-4')
                        self.dp[self.index_param] *=0.95
                        self.loop_number= 1
                    if(self.loop_number == 3):
                        self.params[self.index_param] -= 2*self.dp[self.index_param]
                        #rospy.loginfo('loop 5-3')
			self.loop_number +=1
                    if (self.loop_number ==2):
                        self.dp[self.index_param] *=0.95
                        self.params[self.index_param] += self.dp[self.index_param]
                        #rospy.loginfo('loop 5-2')
                        self.loop_number +=1
                    #rospy.loginfo('dp %s param %s', self.dp[self.index_param], self.params[self.index_param])
                else:
                    #rospy.loginfo('loop 4')
                    self.temp_storage = self.params[self.index_param]
                    #rospy.loginfo('index_param %s param %s ', self.index_param, self.params[self.index_param])
                    self.params[self.index_param] -= 2* self.dp[self.index_param]
                    self.loop_number +=1
                    #rospy.loginfo('index_param %s param %s ', self.index_param, self.params[self.index_param])
        self.iteration +=1
        if(self.iteration %4 == 0):
            if(self.prev_param_index != self.index_param):
                #rospy.loginfo('loop 6')
                self.prev_param_index = self.index_param
                del self.temp_storage2[0:len(self.temp_storage2)]
            else:
                #rospy.loginfo('loop 7')
                self.temp_storage2.append(self.params[self.index_param])
                if(len(self.temp_storage2) > 2):
                    rospy.loginfo('loop 8')
                    self.index_param +=1
                    if self.index_param == 3:
                        self.index_param=0
        self.kp = self.params[0]
        self.ki = self.params[1]
        self.kd = self.params[2]
        #rospy.loginfo('kp: %s, ki: %s, kd: %s', self.kp, self.ki, self.kd)

    def TotalError(self):
        return -1*(self.kp*self.p_error + self.kd*self.d_error + self.ki * self.i_error)

    def get_val(self):
        return -1*(self.kp*self.p_error + self.kd*self.d_error + self.ki*self.i_error)
