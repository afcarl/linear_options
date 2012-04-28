#include <linear_options/SMDPAgent.hh>

#include <fstream>
#include <iomanip>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace rl {

class LOEMAgent : public SMDPAgent
{
public:
    LOEMAgent(rl::state_abstraction& stateAbstraction) : 
        stateAbstraction(&stateAbstraction) {};

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
    void loadOptions(conss std::string& filename);
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

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & options;
    }

    rl::state_abstraction* stateAbstraction;
};

}
