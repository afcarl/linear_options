#include <linear_options/ContinuousRooms.hh>

#include <opencv/highgui.h>

ContinuousRooms::ContinuousRooms(const std::string& filename, double robotRadius, bool randomizeInitialPosition, double safety, Random rng) : 
    map(cv::imread(filename)),
    robotRadius(robotRadius),
    randomPosition(randomizeInitialPosition),
    safetyMargin(safety),
    rng(rng),
    minimaSteps(0)
{
    getCircularROI(robotRadius + safetyMargin, circularROI);
    reset();
    currentState.resize(7);
    updateStateVector();
}

const std::vector<float>& ContinuousRooms::sensation() const
{
    return currentState;
}

void ContinuousRooms::updateStateVector()
{
    // Returns pose and floor color 
    cv::Vec3b intensity = map.at<cv::Vec3b>(y, x);

    // A binary variable represents the color sensed under the robot
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];

    if (red == 0 && green == 255 && blue == 0) {
        currentState[0] = 1;
        currentState[1] = 0;
        currentState[2] = 0;
        currentState[3] = 0;
    } else if (red == 0 && green == 0 && blue == 255) {
        currentState[0] = 0;
        currentState[1] = 1;
        currentState[2] = 0;
        currentState[3] = 0;
    } else if (red == 255 && green == 0 && blue == 255) {
        currentState[0] = 0;
        currentState[1] = 0;
        currentState[2] = 1;
        currentState[3] = 0;
    } else if (red == 255 && green == 255 && blue == 0) {
        currentState[0] = 0;
        currentState[1] = 0;
        currentState[2] = 0;
        currentState[3] = 1;
    }
    
    currentState[4] = x;
    currentState[5] = y;
    currentState[6] = psi;
}

void ContinuousRooms::getCircularROI(int R, std::vector<int>& circularROI)
{
    circularROI.resize(R+1);
    for( int y = 0; y <= R; y++ ) {
        circularROI[y] = cvRound(sqrt((double)R*R - y*y));
    }
}

bool ContinuousRooms::isCollisionFree(double xPrime, double yPrime)
{
   for (int dy = -robotRadius - safetyMargin; dy <= robotRadius + safetyMargin; dy++) { 
       int Rx = circularROI[abs(dy)];
       for (int dx = -Rx; dx <= Rx; dx++ ) { 
           // Check boundary conditions
           if (xPrime + dx >= map.size().width || yPrime + dy >= map.size().height) {
               std::cerr << "Boundary" << std::endl;
               return false;
           }

           // Check if there would be a wall within the circle
           if (map.at<cv::Vec3b>(yPrime + dy, xPrime + dx)[0] == 0 && 
               map.at<cv::Vec3b>(yPrime + dy, xPrime + dx)[1] == 0 && 
               map.at<cv::Vec3b>(yPrime + dy, xPrime + dx)[2] == 0) {
               return false; 
           } 
       }
   }

   return true;
}

bool ContinuousRooms::detectMinima()
{
    if (minimaSteps > MAX_NUMBER_STEPS) {
        std::cout << "**** Max number of steps reached" << std::endl;
        terminated = true;
        return true;
    }

    if (std::sqrt(std::pow(x - lastX, 2) + std::pow(y - lastY, 2)) < MIN_DISPLACEMENT) {
        minimaSteps += 1;
    } else {
        minimaSteps = 0;
    } 

    return false;
}

float ContinuousRooms::apply(int action)
{
   lastX = x;
   lastY = y;

   // Check if we have reached the goal 
   // by entering the bottom right corner yellow room 
   if ((x > map.size().width/2.0 && y > map.size().height/2.0) 
           &&  (map.at<cv::Vec3b>(y, x)[2] == 255 && 
               map.at<cv::Vec3b>(y, x)[1] == 255 && 
               map.at<cv::Vec3b>(y, x)[0] == 0)) {
       terminated = true;
       updateStateVector();
       std::cout << "**** The global goal for the environment was reached" << std::endl;
       return REWARD_SUCCESS;
   } 

   double reward = REWARD_FAILURE;
   if (action == FORWARD) {
       // Moves 1 unit forward in the current orientation
       // with zero mean Gaussian noise with 0.1 std deviation
       double xPrime = x + std::cos(psi) + rng.normal(0.0, 0.1);
       double yPrime = y + std::sin(psi) + rng.normal(0.0, 0.1); 

       if (!isCollisionFree(xPrime, yPrime)) {
           reward = REWARD_FAILURE;
       } else {
           x = xPrime;
           y = yPrime;

           updateStateVector();
           reward = REWARD_FAILURE;  
       }
   } 

   // The left and right actions turn the robot
   // 30 degrees in the specified direction 
   if (action == RIGHT) {
       psi += M_PI/6.0; 
       if (psi == 2.0*M_PI) {
           psi = 0;
       }

       updateStateVector();
       reward = REWARD_FAILURE;  
   }

   if (action == LEFT) {
       psi -= M_PI/6.0;
       if (psi == -2.0 * M_PI) {
           psi = 0;
       }

       updateStateVector();
       reward = REWARD_FAILURE;  
   }

   if (detectMinima()) {
       reward = REWARD_FAILURE_MINIMA; 
   }

   return reward;
}

bool ContinuousRooms::terminal() const
{
    return terminated;
}

void ContinuousRooms::reset()
{
    terminated = false;

    double xInit = 12; 
    double yInit = 12; 

    if (randomPosition) {
        do {
            xInit = rng.uniform(robotRadius/2.0, map.size().width-1);
            yInit = rng.uniform(robotRadius/2.0, map.size().height-1);
        } while (!isCollisionFree(xInit, yInit));
    }

    x = xInit; 
    lastX = x;
    y = yInit; 
    lastY = y;

    psi = M_PI/2.0;
    minimaSteps = 0;
}

int ContinuousRooms::getNumActions()
{
    return NUM_ACTIONS; 
}

void ContinuousRooms::getMinMaxFeatures(std::vector<float> *minFeat,
                                 std::vector<float> *maxFeat)
{

}

void ContinuousRooms::getMinMaxReward(float *minR, float *maxR)
{
    *minR = REWARD_FAILURE;
    *maxR = REWARD_SUCCESS;
}
