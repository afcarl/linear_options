#include <linear_options/SMDPAgent.hh>
#include <linear_options/LOEMModel.hh>

namespace rl {

class LOEMAgent : public SMDPAgent
{
public:

    LOEMAgent();
    ~LOEMAgent();

protected:
    // Linear Expectation Option Model (LOEM)
    // for the behavior policy
    LOEMModel model;
};

}
