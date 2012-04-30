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
     * Choose the best action with probability 1-epsilon, random with epsilon
     * @param phi The current state in the projected n-d space
     * @param The index of the primitive action to execute 
     */
    int epsilonGreedy(const Eigen::VectorXd& phi);

    /**
     * Return the action with the highest return max_o Q(s, O)
     * @param phi The n-dimensional projection of a state
     * @param the LinearOption of maximum value
     */
    LinearOption& getBestOption(const Eigen::VectorXd& phi);

    // Last option executed
    LinearOption& lastOption;

    // Last action executed
    int lastAction;

    // Last state visited
    Eigen::VectorXd lastState;

    LinearOption& currentOption;

    // Maximum value for next state from s
    struct NextStateValueComparator {
        NextStateValueComparator(const Eigen::VectorXd& phi) : phi(phi) {}
        bool operator() (const LinearOption& a, const LinearOption& b) { 
            return a.theta.transpose()*a.F*phi < b.theta.transpose()*b.F*phi;     
        }
        const Eigen::VectorXd& phi;
    };

    // Choose new option according to main behavior policy 
    struct ValueComparator {
        ValueComparator(const Eigen::VectorXd& phi) : phi(phi) {}
        bool operator() (const LinearOption& a, const LinearOption& b) { 
            return a.theta.transpose()*phi < b.theta.transpose()*phi;     
        }
        const Eigen::VectorXd& phi;
    };
};

}
