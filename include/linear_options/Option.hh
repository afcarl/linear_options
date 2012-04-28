#ifndef _OPTION_H_
#define _OPTION_H_
#include <Eigen/Core>
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
};
} // namespace rl

namespace boost {
namespace serialization {

// Serialization for Eigen::VectorXd
template <class Archive>
void save(Archive &ar, const Eigen::VectorXd &cb, unsigned int)
{
    std::size_t nrows = cb.rows();
    ar << nrows;
    for (unsigned int i = 0; i < (unsigned int)cb.rows(); ++i) {
        ar << cb(i);
    }
}

template <class Archive>
void load(Archive &ar, Eigen::VectorXd &cb, unsigned int)
{
    std::size_t rows;
    ar >> rows;
    cb.resize(rows);
    for (unsigned int i = 0; i < (unsigned int)cb.rows(); ++i) {
        ar >> cb(i);
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(Eigen::VectorXd);

// Serialization code for Eigen::MatrixXd
template <class Archive>
void save(Archive &ar, const Eigen::MatrixXd &cb, unsigned int)
{
    std::size_t nrows = cb.rows(), ncols = cb.cols();
    ar << nrows;
    ar << ncols;
    for (unsigned int i = 0; i < (unsigned int)cb.rows(); ++i) {
        for (unsigned int j = 0; j < (unsigned int)cb.cols(); ++j) {
            ar << cb(i,j);
         }
     }
}
template <class Archive>
void load(Archive &ar, Eigen::MatrixXd &cb, unsigned int)
{
    std::size_t rows, cols;
    ar >> rows;
    ar >> cols;
    cb.resize(rows,cols);
    for (unsigned int i = 0; i < (unsigned int)cb.rows(); ++i) {
        for (std::size_t j = 0; j < (unsigned int)cb.cols(); ++j) {
            ar >> cb(i,j);
        }
    }
}
BOOST_SERIALIZATION_SPLIT_FREE(Eigen::MatrixXd);

} // namespace serialization
} // namespace boost


#endif
