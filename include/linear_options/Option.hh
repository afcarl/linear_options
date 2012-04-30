#ifndef _OPTION_H_
#define _OPTION_H_

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
    virtual bool initiate(const Eigen::VectorXd& s) {
        return false;
    }

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
    bool terminate(const Eigen::VectorXd& s) {
        return rng.uniform() < beta(s);
    }

    /**
     * Returns the best action to choose in every state
     * @param phi The current state
     * @return The best action to choose from state phi
     */
    virtual int policy(const Eigen::VectorXd& phi) { return 0; } 

    // The option's parameter vector that we are learning
    Eigen::VectorXd theta;

    // Transition model
    Eigen::MatrixXd F;

    // Reward model 
    Eigen::VectorXd b;

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
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
