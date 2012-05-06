#include <linear_options/StateAbstraction.hh>
#include <linear_options/ContinuousRooms.hh>
#include <linear_options/LinearQ0Learner.hh>
#include <linear_options/RewardDecorator.hh>

#include <opencv/cv.h>
#include <opencv/highgui.h>
/**
 * We build the feature vector from a set of radial basis functions
 * spread over the space in the x, y and psi dimensions. 
 */
struct room_abstraction : public rl::state_abstraction
{
    /**
     * @param U The mean of the RBF
     * @param C
     * @param b
     */
    room_abstraction(Eigen::MatrixXd U, Eigen::Vector3d C, double b) :
       b(b), U(U), C(C.asDiagonal()) {};

    /**
     * @param s Project the input vector in the n-d space
     */
    Eigen::VectorXd operator()(const Eigen::VectorXd& s) {
        Eigen::VectorXd phi(length());
        // The first 4 elements are binary indicator variables for floor color
        phi(0) = s[0];
        phi(1) = s[1];
        phi(2) = s[2];
        phi(3) = s[3];

        // The next 3 elements: x, y, psi
        for (int i = 0; i < U.rows(); i++) {
            phi(i + 4) = b*exp(-0.5*(s.tail(U.cols()) - U.row(i).transpose()).dot(C*(s.tail(U.cols()) - U.row(i).transpose()))); 
            if (phi(i + 4) < 0.1) { 
                phi(i + 4) = 0;
            }
        }

        return phi;
    }

    int length() { return U.rows() + 4; }

private:
    double b;
    Eigen::MatrixXd U;
    Eigen::DiagonalMatrix<double, 3, 3> C;
};

/**
 * Subclasses the LinearOptions to specify the termination set
 */
struct ReachNearestStateOfColor : public rl::LinearOption
{
    ReachNearestStateOfColor(int targetColor) : targetColor(targetColor) {};
    int targetColor;

    /**
     * @Override
     */
    double beta(const Eigen::VectorXd& s) 
    {
        // Check if the bit for target color is set
        if (s[targetColor]) {
            return 1;
        }
        return 0;
    }
};

/**
 * This class acts as a decorator that defines its 
 * own pseudo-reward function over the one returned by 
 * the actual environment. 
 */
struct ReachNearestColorRewardDecorator : public rl::RewardDecorator 
{
    ReachNearestColorRewardDecorator(Agent& agent, int targetColor) : 
        rl::RewardDecorator(agent), 
        targetColor(targetColor) {};

    int targetColor;
    static const double REWARD_SUCCESS = 1;
    static const double REWARD_FAILURE = -0.1;

    double pseudoReward(float reward, const std::vector<float> &s)
    {
        if (s[targetColor]) { 
            return REWARD_SUCCESS;
        } else {
            return REWARD_FAILURE;
        }    
    }

    bool terminal(const std::vector<float> &s)
    {
        if (s[targetColor]) { 
            return true;
        } else {
            return false; 
        }    
    }
};

int main(void)
{
// Radial-basis functions are placed every 10 units in 
// in the x and y dimensions and every 30 degrees
Eigen::MatrixXd U(5200, 3);
int i = 0;
for (double x = 10.2/2.0; x < 200; x += 10) {
    for (double y = 10.2/2.0; y < 200; y += 10) {
        for (double psi = 0; psi <= 360; psi += 30) {
            U(i, 0) = x;
            U(i, 1) = y;
            U(i, 2) = psi;
            i += 1;
        }
    }
}
Eigen::Vector3d C(1.0/10.2, 1.0/10.2, 1/30);
room_abstraction stateAbstraction(U, C, 20);

// Instantiate agents for learning a policy for reaching 
// the subgoals defined by the pseudo-reward functions
std::vector<rl::RewardDecorator*> agents;
for (int color = 0; color < ContinuousRooms::NUM_COLORS; color++) {
    agents.push_back(new ReachNearestColorRewardDecorator(*(new rl::LinearQ0Learner(ContinuousRooms::NUM_ACTIONS, 5e-4, 0.1, 0.6, stateAbstraction)), color));
}

// We use a virtual world of 200x200 units with a 10 units wide robot
const double robotRadius = 5;
ContinuousRooms env("map.png", robotRadius, true);

cv::Mat img = cv::imread("map.png");

// Train a separate agent for each option and take the resulting policy
const unsigned numberLearningEpisodes = 100; 
unsigned agentIdx = 0;
for (auto itAgent = agents.begin(); itAgent != agents.end(); itAgent++) {
    for (unsigned i = 0; i < numberLearningEpisodes; i++) {
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "Agent " << agentIdx << " Episode " << i << std::endl;
        std::cout << "---------------------------------------------" << std::endl;

        auto s = env.sensation();
        auto reward = env.apply((*itAgent)->first_action(s));

            std::cout << "Terminal in environment" << env.terminal() << std::endl;
            std::cout << "Terminal in agent " << (*itAgent)->terminal(s) << std::endl;
        while (!(*itAgent)->terminal(s) && env.terminal() == false) {
            s = env.sensation();
            reward = env.apply((*itAgent)->next_action(reward, s));

            std::cout << "Reward " << reward << std::endl;
            cv::circle(img, cv::Point(s[4], s[5]), 5, cv::Scalar(0, 0, 0), 1); 
            cv::imshow("world", img); 

            if(cv::waitKey(30) >= 0) break;
            sleep(0.25);
        }
    
        s = env.sensation();
        (*itAgent)->last_action(reward);
        env.reset();
    }
    agentIdx += 1;
}

return 0;
}
