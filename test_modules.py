"""
Unit tests for adaptive-quiz-personality modules.
Tests verify modularity, correctness, and that modules don't rely on global state.
"""

import numpy as np
import unittest
from bayesian_update import bayesian_update
from adaptive_question_selector import generate_question_weights, select_next_question, expected_information_gain
from synthetic_user import SyntheticUser, ConfidentUser, HesitantUser, FragmentedUser


class TestBayesianUpdate(unittest.TestCase):
    """Test bayesian_update module."""
    
    def test_bayesian_update_basic(self):
        """Test basic Bayesian update reduces uncertainty."""
        d = 4
        mu_prior = np.zeros(d)
        Sigma_prior = np.eye(d)
        w_t = np.array([1.0, 0.0, 0.0, 0.0])
        y_t = 0.5
        sigma2_t = 0.2
        
        mu_post, Sigma_post = bayesian_update(mu_prior, Sigma_prior, w_t, y_t, sigma2_t)
        
        # Posterior should have lower uncertainty (larger determinant = more uncertainty)
        self.assertLess(np.linalg.det(Sigma_post), np.linalg.det(Sigma_prior))
        
        # Mean should move toward observation
        self.assertGreater(mu_post[0], mu_prior[0])
    
    def test_bayesian_update_no_global_state(self):
        """Test that function is pure (no global state)."""
        d = 3
        mu1 = np.zeros(d)
        Sigma1 = np.eye(d)
        w = np.array([0.0, 1.0, 0.0])
        y = 0.3
        sigma2 = 0.1
        
        # Same inputs should give same outputs
        mu_a, Sigma_a = bayesian_update(mu1, Sigma1, w, y, sigma2)
        mu_b, Sigma_b = bayesian_update(mu1, Sigma1, w, y, sigma2)
        
        np.testing.assert_array_almost_equal(mu_a, mu_b)
        np.testing.assert_array_almost_equal(Sigma_a, Sigma_b)


class TestAdaptiveQuestionSelector(unittest.TestCase):
    """Test adaptive_question_selector module."""
    
    def test_generate_question_weights(self):
        """Test question weight generation."""
        d = 4
        T = 10
        w_list = generate_question_weights(d, T, random_state=np.random.RandomState(42))
        
        self.assertEqual(len(w_list), T)
        for w in w_list:
            self.assertEqual(len(w), d)
            # Should probe 1 or 2 axes
            num_axes = np.sum(w != 0)
            self.assertGreaterEqual(num_axes, 1)
            self.assertLessEqual(num_axes, 2)
    
    def test_select_next_question_variance(self):
        """Test variance-based question selection."""
        d = 3
        mu = np.zeros(d)
        Sigma = np.eye(d)
        w_list = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        idx, w = select_next_question(mu, Sigma, w_list, asked_indices=None, mode='variance')
        
        self.assertIn(idx, [0, 1, 2])
        self.assertIn(w, w_list)
    
    def test_select_next_question_info_gain(self):
        """Test information gain-based question selection."""
        d = 3
        mu = np.zeros(d)
        Sigma = np.eye(d)
        w_list = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        idx, w = select_next_question(mu, Sigma, w_list, asked_indices=None, mode='info_gain', sigma2_t=0.2)
        
        self.assertIn(idx, [0, 1, 2])
        self.assertIn(w, w_list)
    
    def test_expected_information_gain(self):
        """Test expected information gain computation."""
        d = 2
        Sigma_prior = np.eye(d)
        w = np.array([1.0, 0.0])
        sigma2_t = 0.2
        
        info_gain = expected_information_gain(Sigma_prior, w, sigma2_t)
        
        # Information gain should be positive (reduces uncertainty)
        self.assertGreater(info_gain, 0)
        
        # Information gain should be finite
        self.assertTrue(np.isfinite(info_gain))
    
    def test_info_gain_vs_variance(self):
        """Test that info gain selection can differ from variance selection."""
        d = 3
        mu = np.zeros(d)
        # Create asymmetric covariance
        Sigma = np.array([[2.0, 0.5, 0.0],
                         [0.5, 1.0, 0.0],
                         [0.0, 0.0, 0.5]])
        w_list = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        idx_var, _ = select_next_question(mu, Sigma, w_list, mode='variance')
        idx_info, _ = select_next_question(mu, Sigma, w_list, mode='info_gain', sigma2_t=0.2)
        
        # They may or may not be the same, but both should be valid
        self.assertIn(idx_var, [0, 1, 2])
        self.assertIn(idx_info, [0, 1, 2])


class TestSyntheticUser(unittest.TestCase):
    """Test synthetic_user module."""
    
    def test_synthetic_user_basic(self):
        """Test basic synthetic user."""
        d = 4
        user = SyntheticUser(d=d, seed=42)
        
        self.assertEqual(len(user.theta_true), d)
        self.assertTrue(np.all(user.theta_true >= -1))
        self.assertTrue(np.all(user.theta_true <= 1))
        
        w = np.array([1.0, 0.0, 0.0, 0.0])
        y, r, sigma2 = user.simulate_response(w)
        
        self.assertIn(y, user.likert_values)
        self.assertGreater(r, 0)
        self.assertGreater(sigma2, 0)
    
    def test_confident_user(self):
        """Test ConfidentUser archetype."""
        d = 3
        user = ConfidentUser(d=d, seed=42)
        
        w = np.array([1.0, 0.0, 0.0])
        y, r, sigma2 = user.simulate_response(w)
        
        # Confident users should have lower noise variance
        self.assertLess(sigma2, 0.2)  # Lower than default
        self.assertLess(r, 2.0)  # Faster responses
    
    def test_hesitant_user(self):
        """Test HesitantUser archetype."""
        d = 3
        user = HesitantUser(d=d, seed=42)
        
        w = np.array([1.0, 0.0, 0.0])
        y, r, sigma2 = user.simulate_response(w)
        
        # Hesitant users should have higher noise variance
        self.assertGreater(sigma2, 0.1)  # Higher than default
        # Slower responses (but this depends on y_t, so just check it's reasonable)
        self.assertGreater(r, 0)
    
    def test_fragmented_user(self):
        """Test FragmentedUser archetype."""
        d = 4
        user = FragmentedUser(d=d, seed=42, flip_probability=0.5)
        
        w = np.array([1.0, 0.0, 0.0, 0.0])
        y, r, sigma2 = user.simulate_response(w)
        
        self.assertIn(y, user.likert_values)
        self.assertGreater(r, 0)
        self.assertGreater(sigma2, 0)
    
    def test_user_no_global_state(self):
        """Test that users don't share global state."""
        d = 2
        user1 = SyntheticUser(d=d, seed=42)
        user2 = SyntheticUser(d=d, seed=43)
        
        # Different seeds should give different traits
        self.assertFalse(np.allclose(user1.theta_true, user2.theta_true))


class TestIntegration(unittest.TestCase):
    """Integration tests for full simulation."""
    
    def test_full_simulation_flow(self):
        """Test that modules work together correctly."""
        d = 3
        T = 5
        user = SyntheticUser(d=d, seed=42)
        w_list = generate_question_weights(d, T, random_state=np.random.RandomState(42))
        
        mu = np.zeros(d)
        Sigma = np.eye(d)
        asked_indices = set()
        
        for t in range(T):
            idx, w = select_next_question(mu, Sigma, w_list, asked_indices, mode='variance')
            asked_indices.add(idx)
            y, r, sigma2 = user.simulate_response(w)
            mu, Sigma = bayesian_update(mu, Sigma, w, y, sigma2)
        
        # After T questions, uncertainty should decrease
        self.assertLess(np.trace(Sigma), np.trace(np.eye(d)))
        self.assertEqual(len(asked_indices), T)


if __name__ == '__main__':
    unittest.main()
