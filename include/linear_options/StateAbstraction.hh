#ifndef __STATE_ABSTRACTION_H__ 
#define __STATE_ABSTRACTION_H__

#include <Eigen/Core>

namespace rl {
    struct state_abstraction : public std::unary_function<Eigen::VectorXd, Eigen::VectorXd> 
    {
        Eigen::VectorXd operator()(const Eigen::VectorXd& s) { return s; };
    };

    struct no_abstraction : public state_abstraction
    {
        Eigen::VectorXd operator()(const Eigen::VectorXd& s) { return s; };
    };
}

#endif
