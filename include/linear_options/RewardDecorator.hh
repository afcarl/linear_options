#ifndef __REWARD_DECORATOR_H__
#define __REWARD_DECORATOR_H__

namespace rl {

/**
 * When learning a policy for an option, we need to define a pseudo-reward function
 * to reach a given subgoal. This abstract class allows to hide this transformation
 * from a regular learning agent. 
 */
class RewardDecorator : public Agent 
{
public:
    /**
     * @param agent The agent that we wish to shield from the actual reward function.
     */
    RewardDecorator(Agent& agent) : agent(&agent) {};
    virtual ~RewardDecorator() {};

    /**
     * @Override
     */
    int first_action(const std::vector<float> &s) { return agent->first_action(s); }

    /**
     * @Override
     */
    int next_action(float r, const std::vector<float> &s) { 
        return agent->next_action(pseudoReward(r, s), s); 
    }

    /**
     * @Override
     */
    void last_action(float r) { agent->last_action(r); }

    /**
     * @Override
     */
    void setDebug(bool d) { agent->setDebug(d); }

    /**
     * This function overrides the actual global reward defined 
     * for the given task in the environment. 
     * @param phi The actual state
     */
    virtual double pseudoReward(float r, const std::vector<float> &s) = 0;
  
    /**
     * This function allows for the last_action function of the 
     * learning agent to be called at the appropriate moment. 
     * @param s The current state
     */ 
    virtual bool terminal(const std::vector<float>& s) = 0;

private:
    Agent* agent;
};
} // namespace rl
#endif
