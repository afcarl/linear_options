#ifndef __STATE_ABSTRACTION_H__ 
#define __STATE_ABSTRACTION_H__

#include <Eigen/Core>

namespace rl {
    struct state_abstraction : public std::unary_function<Eigen::VectorXd, Eigen::VectorXd> 
    {
        virtual Eigen::VectorXd operator()(const Eigen::VectorXd& s) = 0;
    };

    struct no_abstraction : public state_abstraction
    {
        Eigen::VectorXd operator()(const Eigen::VectorXd& s) { return s; };
    };
}

#endif
