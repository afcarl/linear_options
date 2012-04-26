#include <linear_options/LOEMAgent.hh>

namespace rl {

/**
 * Implements the agent described in the paper
 * "Linear Options" that interleaves one step of 
 * planing update with one step of intra-option learning
 * for all the options with their attached LOEM model.
 */
class DynaLOEMAgent : public LOEMAgent
{
public:
    /**
     * @Override
     */
    int first_action(const std::vector<float> &s);

    /**
     * @Override
     */
    int next_action(float r, const std::vector<float> &s);

    /**
     * @Override
     */
    void last_action(float r);

    /**
     * @Override
     */
    void setDebug(bool d);

    DynaLOEMAgent();

    ~DynaLOEMAgent();

protected:    
    // We maintain one LOEM model for every option
    std::vector<LOEMModel> optionModels;  

    // Last action executed
    int lastAction;
};

}
