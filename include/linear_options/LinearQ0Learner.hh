#ifndef __LINEAR_Q0_LEARNER_H__
#define __LINEAR_Q0_LEARNER_H__

#include <linear_options/LOEMAgent.hh>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace rl {

/**
 * An option learner is a specialized agent 
 * which defines a pseudo-reward function over the
 * existing environmental feedback. 
 *
 * A pseudo-q-function is learnt. 
 * TODO Extract an interface. 
 */
class LinearQ0Learner : public Agent 
{
public:
    LinearQ0Learner(unsigned numActions, double alpha, double epsilon, double gamma, rl::state_abstraction& stateAbstraction, Random rng = Random()) : 
        numActions(numActions), 
        alpha(alpha),
        epsilon(epsilon),
        gamma(gamma),
        stateAbstraction(&stateAbstraction),
        rng(rng)
    { actionValueThetas.resize(numActions); }
    virtual ~LinearQ0Learner() {};

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
     * Return the best action to take with respect to the current theta estimates
     * for every action. 
     * @param phi The current n-d state
     * @return The action corresponding to the greedy policy
     */
    int getBestAction(const Eigen::VectorXd& phi);

    /**
     * Convert an STL vector to an Eigen::Vector of double.
     * @param s The vector to convert
     * @return The Eigen::Vector representation.
     * FIXME Duplicate code. 
     */
    Eigen::VectorXd convertVector(const std::vector<float>& s) {
        Eigen::VectorXd out(s.size());
        for (unsigned i = 0; i < s.size(); i++) {
            out(i) = s[i];
        }
        return out; 
    }

    /**
     * Project the input state into a higher dimensional space
     * using the pre-defined state abstraction function. 
     */
    inline Eigen::VectorXd project(const std::vector<float>& s) 
    {
        return (*stateAbstraction)(convertVector(s));  
    }

    // Serialization for model parameters 
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & actionValueThetas;
    }

    // Linear approximation for the pseudo-Q-function. 
    // Used by the option's policy for control
    std::vector<Eigen::VectorXd> actionValueThetas;

    unsigned numActions;
    double alpha;
    double epsilon;
    double gamma;
    rl::state_abstraction* stateAbstraction;
    Random rng;

private:
    // Last primitive action executed during learning
    int lastAction;
   
    // Last state visited 
    Eigen::VectorXd lastPhi; 
};
} // namespace rl

#endif
