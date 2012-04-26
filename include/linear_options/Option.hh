#ifndef _OPTION_H_
#define _OPTION_H_
#include <Eigen/Core>

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
    /**
     * @param s The n-dimensional feature vector. 
     * @return True if the option can be taken state s
     * @FIXME Assumes options are available in all states
     */
    virtual bool initiate(const Eigen::VectorXd& s) {
        return true;
    }

    /**
     * @param s The n-dimensional feature vector. 
     * @return The probability of termination given a feature vector
     */
    virtual double terminate(const Eigen::VectorXd& s);

    // The option's parameter vector that we are learning
    Eigen::VectorXd theta;
};

}

#endif
