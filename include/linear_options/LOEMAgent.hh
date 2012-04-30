#include <linear_options/SMDPAgent.hh>
#include <linear_options/StateAbstraction.hh>
#include <linear_options/Option.hh>

#include <fstream>
#include <iomanip>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace rl {

class LOEMAgent : public SMDPAgent
{
public:
    LOEMAgent(unsigned numActions, double alpha, double epsilon, double gamma, rl::state_abstraction& stateAbstraction, Random rng) : 
        numActions(numActions), 
        alpha(alpha),
        epsilon(epsilon),
        gamma(gamma),
        stateAbstraction(&stateAbstraction),
        rng(rng)
    {};
    virtual ~LOEMAgent() {};

    /**
     * @Override
     */
    void saveOptions(const std::string& filename) 
    {
        std::ofstream file(filename); 
        boost::archive::text_oarchive oa(file);
        oa << options;
    }

    /**
     * @Override
     */
    void loadOptions(const std::string& filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        boost::archive::text_iarchive ia(ifs);
        ia >> options;
    }

protected:
    /**
     * Project the input state into a higher dimensional space
     * using the pre-defined state abstraction function. 
     */
    inline Eigen::VectorXd project(const std::vector<float>& s) 
    {
        return (*stateAbstraction)(convertVector(s));  
    }

    // Contains the linear options loaded from disk
    std::vector<LinearOption> options;

    unsigned numActions;
    double alpha;
    double epsilon;
    double gamma;
    rl::state_abstraction* stateAbstraction;
    Random rng;

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & options;
    }

};

}
