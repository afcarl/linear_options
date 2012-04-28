#include <linear_options/LOEMAgent.hh>

namespace rl {

/**
 * Implements the agent described in the paper
 * "Linear Options" that interleaves one step of 
 * planing update with one step of intra-option learning
 * for all the options with their attached LOEM model.
 */
class DynaLOEMAgent : public LOEMAgent
{
public:
    DynaLOEMAgent();

    ~DynaLOEMAgent();

    /**
     * @Override
     */
    int first_action(const std::vector<float> &s);

    /**
     * @Override
     */
    int next_action(float r, const std::vector<float> &s);

    /**
     * @Override
     */
    void last_action(float r);

    /**
     * @Override
     */
    void setDebug(bool d);

protected:    
    /**
     * Return the action with the highest return max_o Q(s, O)
     * @param phi The n-dimensional projection of a state
     * @param the LinearOption of maximum value
     */
    LinearOption& getBestAction(const Eigen::VectorXd phi);

    // Last action executed
    LinearOption& lastAction;

    Eigen::VectorXd lastState;

    // Last state visited
    Eigen::VectorXd lastState;
};

}
