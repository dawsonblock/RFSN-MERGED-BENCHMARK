from rfsn_upstream.learner.contextual_bandit import ThompsonBandit
from rfsn_upstream.learner.reward import reward_from_episode
import json
import glob

bandit = ThompsonBandit()

for path in glob.glob("logs/episodes/*.json"):
    with open(path) as f:
        ep = json.load(f)
    r = reward_from_episode(ep)
    bandit.update(ep["arm_key"], r)

print("Learner updated (offline)")
