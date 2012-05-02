#ifndef __OPTION_H__
#define __OPTION_H__

#include <linear_options/serialization.hh>

#include <limits>
#include <Eigen/Core>
#include <rl_common/Random.h>

namespace rl {

/**
 * Tagging interface
 */
struct Option {}; 

/**
 * A linear option is an extension for the
 * options framework from the tabular representation
 * to a more general linear form.
 *
 * The quantities are defined over the n-dimensional feature
 * space rather than over the states directly.
 */
struct LinearOption : public Option
{
    LinearOption() : rng(Random()) {};

    /**
     * @param s The n-dimensional feature vector. 
     * @return True if the option can be taken state s
     * @FIXME Assumes options are available in all states
     */
    virtual bool initiate(const Eigen::VectorXd& s) { return true; }

    /**
     * @param s The n-dimensional feature vector. 
     * @return The probability of termination given a feature vector
     */
    virtual double beta(const Eigen::VectorXd& s) { return 1; }

    /**
     * Indicate if the option should terminate in the current state
     * @param s The n-dimensional feature vector. 
     * @return True if the execution of the option must stop, false otherwise.
     */
    bool terminate(const Eigen::VectorXd& s) { return rng.uniform() < beta(s); }

    /**
     * Returns the best action to choose in every state
     * @param phi The current state
     * @return The best action to choose from state phi
     */
    int greedyPolicy(const Eigen::VectorXd& phi) { 
        double maxValue;
        double maxAction = -1*std::numeric_limits<double>::max();
        for (unsigned i = 0; i < actionValueThetas.size(); i++) {
           if (actionValueThetas[i].dot(phi) > maxValue) {
               maxAction = i;
           }
        }

        return maxAction;
    }

    // The option's parameter vector that we are learning. 
    // Used by the behavior policy for control
    Eigen::VectorXd theta;

private:
    // Linear approximation for the pseudo-Q-function. 
    // Used by the option's policy for control
    std::vector<Eigen::VectorXd> actionValueThetas;

    // Serialization for model parameters 
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & actionValueThetas;
        ar & theta;
    }

    Random rng;
};

/**
 * A model can be associated with a linear option.
 * The learning algorithm is implemented in the derived classes.
 */
struct LinearOptionModel
{
    // Transition model
    Eigen::MatrixXd F;

    // Reward model 
    Eigen::VectorXd b;

private:
    // Serialization for model parameters 
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & F;
        ar & b;
    }
};

} // namespace rl

#endif
