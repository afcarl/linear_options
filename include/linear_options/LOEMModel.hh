#include <Eigen/Core>

namespace rl {
/**
 * A Linear Expectation Option Model (LOEM)
 * has two components:
 * 1. F, predicting the next state 
 * 2. b, predicting the reward
 *
 * It uses least-squares regression to estimate them. 
 *
 * TODO MDPModel is not general enough for subclassing. 
 */
namespace rl {

class LOEMModel
{  
public:
   /** Update the MDP model with a single experience. */
   bool updateWithExperience(experience &instance);

   /**
    * Transition model 
    * @param s The n-dimensional state 
    * @return The predicted transition
    */
   Eigen::VectorXd predictNextState(Eigen::VectorXd& s);

   /**
    * Reward model 
    * @param s The n-dimensional state 
    * @return The predicted reward 
    */
    double Eigen::VectorXd predictReward(Eigen::VectorXd& s); 

    LOEMModel();
    ~LOEMModel(); 

private:
    // Transition model
    EigenMatrixXd F;

    // Reward model
    EigenVectorXd b;
};

}
