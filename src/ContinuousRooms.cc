#include <linear_options/ContinuousRooms.hh>

#include <opencv/highgui.h>

ContinuousRooms::ContinuousRooms(const std::string& filename, double robotRadius, double scaling, Random rng) : 
    map(cv::imread(filename)),
    robotRadius(robotRadius),
    scaling(scaling),
    x(3),
    y(8),
    psi(M_PI/2.0),
    terminated(false),
    rng(rng)
{
    currentState.resize(7);
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

void ContinuousRooms::getCircularROI(int R, std::vector<int>& RxV)
{
    RxV.resize(R+1);
    for( int y = 0; y <= R; y++ ) {
        RxV[y] = cvRound(sqrt((double)R*R - y*y));
    }
}

float ContinuousRooms::apply(int action)
{
   if (action == FORWARD) {
       // Moves 1 unit forward in the current orientation
       // with zero mean Gaussian noise with 0.1 std deviation
       double xPrime = x + std::cos(psi) + rng.normal(0.0, 0.1);
       double yPrime = y + std::sin(psi) + rng.normal(0.0, 0.1); 

       // Detect collision 
       // Pretend the robot is a square     
       std::vector<int> RxV;
       getCircularROI(robotRadius, RxV);

       for(int dy = -robotRadius; dy <= robotRadius; dy++) { 
           int Rx = RxV[abs(dy)];
           for( int dx = -Rx; dx <= Rx; dx++ ) { 
               // Check if there would be a wall within the circle
               if (map.at<cv::Vec3b>(yPrime + dy, xPrime + dx)[0] == 0 && 
                   map.at<cv::Vec3b>(yPrime + dy, xPrime + dx)[1] == 0 && 
                   map.at<cv::Vec3b>(yPrime + dy, xPrime + dx)[2] == 0) {
                   return REWARD_FAILURE;  
               }
           }
       }

       x = xPrime;
       y = yPrime;

       updateStateVector();
       return REWARD_FAILURE;  
   } 

   // The left and right actions turn the robot 30 degrees
   // in the specified direction 
   if (action == LEFT) {
       psi += M_PI/6.0; 
       if (psi == 2.0*M_PI) {
           psi = 0;
       }

       updateStateVector();
       return REWARD_FAILURE;  
   }

   if (action == RIGHT) {
       psi -= M_PI/6.0;
       if (psi == -2.0 * M_PI) {
           psi = 0;
       }

       updateStateVector();
       return REWARD_FAILURE;  
   }

   // Check if we have reached the goal 
   // by entering the bottom right corner yellow room 
   cv::Vec3b intensity = map.at<cv::Vec3b>(y, x);
   uchar blue = intensity.val[0];
   uchar green = intensity.val[1];
   uchar red = intensity.val[2];

   if ((x > map.size().width/2.0 && y > map.size().height/2.0) 
           &&  (red == 255 && green == 255 && blue == 0)) {
       terminated = true;
       
       updateStateVector();
       return REWARD_SUCCESS;
   } 

   return REWARD_FAILURE;   
}

bool ContinuousRooms::terminal() const
{
    return terminated;
}

void ContinuousRooms::reset()
{
    terminated = false;
    x = 3;
    y = 8;
    psi = M_PI/2.0;
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