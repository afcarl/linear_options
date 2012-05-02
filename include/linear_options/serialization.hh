#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#include <Eigen/Core>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

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
