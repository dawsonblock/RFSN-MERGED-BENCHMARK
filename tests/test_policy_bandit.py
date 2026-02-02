"""Tests for the policy_bandit module."""


from rfsn_controller.policy_bandit import (
    BanditArm,
    ContextFeatures,
    ThompsonBandit,
    create_policy,
)


class TestBanditArm:
    """Tests for BanditArm class."""
    
    def test_initial_state(self) -> None:
        """Test initial arm state."""
        arm = BanditArm(name="test")
        
        assert arm.pulls == 0
        assert arm.successes == 0
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
    
    def test_update_success(self) -> None:
        """Test updating arm with success."""
        arm = BanditArm(name="test")
        arm.update(1.0)
        
        assert arm.pulls == 1
        assert arm.successes == 1
        assert arm.alpha > 1.0
    
    def test_update_failure(self) -> None:
        """Test updating arm with failure."""
        arm = BanditArm(name="test")
        arm.update(0.0)
        
        assert arm.pulls == 1
        assert arm.successes == 0
        assert arm.beta > 1.0
    
    def test_sample_bounds(self) -> None:
        """Test that samples are in [0, 1]."""
        arm = BanditArm(name="test")
        import random
        rng = random.Random(42)
        
        for _ in range(100):
            sample = arm.sample(rng)
            assert 0.0 <= sample <= 1.0
    
    def test_mean(self) -> None:
        """Test mean calculation."""
        arm = BanditArm(name="test", alpha=3.0, beta=2.0)
        
        # Mean of Beta(3, 2) = 3 / (3 + 2) = 0.6
        assert abs(arm.mean() - 0.6) < 0.01
    
    def test_to_dict_and_back(self) -> None:
        """Test serialization."""
        arm = BanditArm(name="test", pulls=5, successes=3, alpha=4.0, beta=3.0)
        
        d = arm.to_dict()
        restored = BanditArm.from_dict(d)
        
        assert restored.name == arm.name
        assert restored.pulls == arm.pulls
        assert restored.alpha == arm.alpha


class TestThompsonBandit:
    """Tests for ThompsonBandit class."""
    
    def test_deterministic_with_seed(self) -> None:
        """Test that same seed gives same choices."""
        bandit1 = ThompsonBandit(seed=42)
        bandit2 = ThompsonBandit(seed=42)
        
        choice1 = bandit1.choose()
        choice2 = bandit2.choose()
        
        assert choice1 == choice2
    
    def test_different_seeds_differ(self) -> None:
        """Test that different seeds give different sequences."""
        bandit1 = ThompsonBandit(seed=42)
        bandit2 = ThompsonBandit(seed=43)
        
        # Collect multiple choices
        choices1 = [bandit1.choose() for _ in range(10)]
        choices2 = [bandit2.choose() for _ in range(10)]
        
        # Should differ at some point (probabilistic, but very likely)
        assert choices1 != choices2
    
    def test_choose_from_available(self) -> None:
        """Test choosing from limited options."""
        bandit = ThompsonBandit(seed=42)
        
        choice = bandit.choose(available=["action_a", "action_b"])
        
        assert choice in ["action_a", "action_b"]
    
    def test_choose_top_k(self) -> None:
        """Test choosing top k arms."""
        bandit = ThompsonBandit(seed=42)
        
        top_3 = bandit.choose_top_k(3)
        
        assert len(top_3) == 3
        assert len(set(top_3)) == 3  # All unique
    
    def test_update_affects_choice(self) -> None:
        """Test that updates affect future choices."""
        bandit = ThompsonBandit(seed=42, arm_names=["a", "b"])
        
        # Heavily reward arm "a"
        for _ in range(100):
            bandit.update("a", 1.0)
            bandit.update("b", 0.0)
        
        # Now "a" should almost always be chosen
        choices = [bandit.choose(["a", "b"]) for _ in range(10)]
        
        assert choices.count("a") > choices.count("b")
    
    def test_get_stats(self) -> None:
        """Test getting arm statistics."""
        bandit = ThompsonBandit(seed=42, arm_names=["a", "b"])
        bandit.update("a", 1.0)
        bandit.update("b", 0.0)
        
        stats = bandit.get_stats()
        
        assert "a" in stats
        assert "b" in stats
        assert stats["a"]["pulls"] == 1
        assert stats["a"]["successes"] == 1
    
    def test_save_and_load(self, tmp_path) -> None:
        """Test persistence to SQLite."""
        db_path = str(tmp_path / "policy.db")
        
        # Create and train bandit
        bandit1 = ThompsonBandit(seed=42, arm_names=["a", "b"])
        for _ in range(10):
            bandit1.update("a", 1.0)
        bandit1.save(db_path)
        
        # Load into new bandit
        bandit2 = ThompsonBandit(seed=42, arm_names=["a", "b"])
        bandit2.load(db_path)
        
        assert bandit2.arms["a"].pulls == 10
        assert bandit2.arms["a"].successes == 10
    
    def test_to_json_and_back(self) -> None:
        """Test JSON serialization."""
        bandit = ThompsonBandit(seed=42, arm_names=["a", "b"])
        bandit.update("a", 1.0)
        
        json_data = bandit.to_json()
        restored = ThompsonBandit.from_json(json_data)
        
        assert restored.seed == bandit.seed
        assert restored.arms["a"].pulls == bandit.arms["a"].pulls


class TestContextFeatures:
    """Tests for ContextFeatures class."""
    
    def test_to_vector(self) -> None:
        """Test feature vector conversion."""
        features = ContextFeatures(
            feature_mode="repair",
            error_type="assertion",
            repo_size="medium",
            language="python",
            last_outcome="success",
            step_number=6,
        )
        
        vector = features.to_vector()
        
        assert len(vector) == 6
        assert all(0.0 <= v <= 1.0 for v in vector)


class TestCreatePolicy:
    """Tests for create_policy helper."""
    
    def test_create_without_db(self) -> None:
        """Test creating policy without persistence."""
        policy = create_policy(seed=42)
        
        assert isinstance(policy, ThompsonBandit)
        assert policy.seed == 42
    
    def test_create_with_db(self, tmp_path) -> None:
        """Test creating policy with persistence."""
        db_path = str(tmp_path / "policy.db")
        
        policy = create_policy(db_path=db_path, seed=42)
        policy.update("test_arm", 1.0)
        policy.save(db_path)
        
        # Load again
        policy2 = create_policy(db_path=db_path, seed=42)
        
        assert "test_arm" in policy2.arms
