#ifndef _OPTION_H_
#define _OPTION_H_

#include <limits>
#include <Eigen/Core>
#include <rl_common/Random.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

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

    /**
     * Control with function approximation and Q(0) 
     */
    int next_action(double reward, const Eigen::VectorXd& phiPrime) {
        
        actionValueThetas[lastAction] = actionValueThetas[lastAction].array() + lastPhi.array()*(reward + gamma*actionValueThetas[greedyPolicy(phiPrime)].dot(phiPrime) - actionValueThetas[lastAction].dot(lastPhi))*alpha;

        lastAction  = (rng.uniform() < epsilon) ? rng.uniformDiscrete(0, numActions-1) : greedyPolicy(phiPrime); 
        lastPhi = phiPrime;

        return lastAction;
    }

    // The option's parameter vector that we are learning. 
    // Used by the behavior policy for control
    Eigen::VectorXd theta;

    // Transition model
    Eigen::MatrixXd F;

    // Reward model 
    Eigen::VectorXd b;

private:
    // Linear approximation for the pseudo-Q-function. 
    // Used by the option's policy for control
    std::vector<Eigen::VectorXd> actionValueThetas;

    // Last primitive action executed during learning
    int lastAction;
   
    // Last state visited 
    Eigen::VectorXd lastPhi; 

    double alpha;

    double epsilon;

    double gamma;

    int numActions;

    // Serialization for model parameters 
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & actionValueThetas;
        ar & theta;
        ar & F;
        ar & b;
    }

    Random rng;
};
} // namespace rl

namespace boost {
namespace serialization {
// MatrixXd
template<class Archive>
void load( Archive & ar,
           Eigen::MatrixXd & t,
           const unsigned int file_version )
{
    int n0;
    ar >> BOOST_SERIALIZATION_NVP(n0);
    int n1;
    ar >> BOOST_SERIALIZATION_NVP(n1);
    t.resize( n0, n1 );
    ar >> make_array(t.data(), t.rows()*t.cols());
}
template<typename Archive>
void save( Archive & ar,
           const Eigen::MatrixXd & t,
           const unsigned int file_version )
{
    int n0 = t.rows();
    ar << BOOST_SERIALIZATION_NVP(n0);
    int n1 = t.cols();
    ar << BOOST_SERIALIZATION_NVP(n1);
    ar << boost::serialization::make_array(t.data(),
                                           t.rows()*t.cols());
}
template<class Archive>
void serialize( Archive & ar,
                Eigen::MatrixXd& t,
                const unsigned int file_version )
{
    split_free(ar, t, file_version);
}

// Eigen::VectorXd
template<class Archive>
void load( Archive & ar,
           Eigen::VectorXd & t,
           const unsigned int file_version )
{
    int n0;
    ar >> BOOST_SERIALIZATION_NVP(n0);
    t.resize( n0 );
    ar >> make_array(t.data(), t.size());
}
template<typename Archive>
void save( Archive & ar,
           const Eigen::VectorXd & t,
           const unsigned int file_version )
{
    int n0 = t.size();
    ar << BOOST_SERIALIZATION_NVP(n0);
    ar << boost::serialization::make_array(t.data(),
                                           t.size());
}
template<class Archive>
void serialize( Archive & ar,
                Eigen::VectorXd& t,
                const unsigned int file_version )
{
    split_free(ar, t, file_version);
}

} // namespace serialization
} // namespace boost
#endif
