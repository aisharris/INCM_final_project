import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import unittest
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd

@dataclass
class TestResult:
    """Store test results for analysis"""
    name: str
    success: bool
    metrics: Dict
    description: str

class DFTModelTester:
    def __init__(self, model):
        self.model = model
        self.results: List[TestResult] = []
        
    def run_all_tests(self) -> List[TestResult]:
        """Run all model tests and return results"""
        self.test_basic_functionality()
        self.test_stability()
        self.test_noise_robustness()
        self.test_adaptation()
        self.test_memory_effects()
        return self.results
    
    def test_basic_functionality(self):
        """Test basic model properties and response to inputs"""
        # Reset model
        self.model.__init__()
        
        # Test 1: Single target response
        target_pos = 0
        results, times = self.model.prepare_movement(target_pos, 50)
        
        # Analyze results
        final_peak = np.argmax(results[-1])
        peak_value = np.max(results[-1])
        expected_peak = self.model.field_size // 2  # Center position
        
        metrics = {
            'peak_position_error': abs(final_peak - expected_peak),
            'peak_amplitude': peak_value,
            'convergence_time': self._calculate_convergence_time(results)
        }
        
        success = (metrics['peak_position_error'] < self.model.field_size * 0.1 and 
                  metrics['peak_amplitude'] > 0)
        
        self.results.append(TestResult(
            name="Basic Functionality",
            success=success,
            metrics=metrics,
            description="Tests basic response to single target input"
        ))
        
    def test_stability(self):
        """Test stability of prepared movements"""
        self.model.__init__()
        
        # Test sustained activation
        target_pos = 0
        simulation_time = 100
        results, times = self.model.prepare_movement(target_pos, simulation_time)
        
        # Analyze stability
        last_quarter = results[-int(len(results)/4):]
        peak_positions = np.argmax(last_quarter, axis=1)
        peak_values = np.max(last_quarter, axis=1)
        
        metrics = {
            'position_std': np.std(peak_positions),
            'amplitude_std': np.std(peak_values),
            'mean_position': np.mean(peak_positions),
            'mean_amplitude': np.mean(peak_values)
        }
        
        success = (metrics['position_std'] < 2 and  # Less than 2 units drift
                  metrics['amplitude_std'] < 0.5)    # Stable amplitude
                  
        self.results.append(TestResult(
            name="Stability Analysis",
            success=success,
            metrics=metrics,
            description="Tests stability of prepared movement"
        ))
        
    def test_noise_robustness(self):
        """Test model's robustness to different noise levels"""
        original_noise = self.model.noise_strength
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        stability_measures = []
        
        for noise in noise_levels:
            self.model.__init__()
            self.model.noise_strength = noise
            
            results, _ = self.model.prepare_movement(0, 50)
            peak_positions = np.argmax(results[-20:], axis=1)
            stability_measures.append(np.std(peak_positions))
        
        # Restore original noise
        self.model.noise_strength = original_noise
        
        metrics = {
            'noise_levels': noise_levels,
            'stability_measures': stability_measures,
            'slope': np.polyfit(noise_levels, stability_measures, 1)[0]
        }
        
        success = metrics['slope'] < 10  # Reasonable degradation with noise
        
        self.results.append(TestResult(
            name="Noise Robustness",
            success=success,
            metrics=metrics,
            description="Tests model behavior under different noise levels"
        ))
        
    def test_adaptation(self):
        """Test adaptation mechanisms"""
        self.model.__init__()
        
        # Test repeated movements
        n_repetitions = 5
        adaptation_effects = []
        
        for i in range(n_repetitions):
            results, _ = self.model.prepare_movement(0, 20)
            peak_timing = self._calculate_convergence_time(results)
            adaptation_effects.append(peak_timing)
        
        metrics = {
            'adaptation_trend': np.polyfit(range(n_repetitions), 
                                         adaptation_effects, 1)[0],
            'final_adaptation': self.model.adaptation.max(),
            'adaptation_times': adaptation_effects
        }
        
        success = metrics['adaptation_trend'] < 0  # Should get faster
        
        self.results.append(TestResult(
            name="Adaptation Analysis",
            success=success,
            metrics=metrics,
            description="Tests adaptation effects on movement preparation"
        ))
        
    def test_memory_effects(self):
        """Test memory trace effects"""
        self.model.__init__()
        
        # Prepare sequence of movements
        positions = [0, 20, 0, 20]
        memory_effects = []
        
        for pos in positions:
            results, _ = self.model.prepare_movement(pos, 30)
            memory_effects.append(self.model.memory_trace.copy())
        
        metrics = {
            'memory_correlation': np.corrcoef(memory_effects[-2], 
                                            memory_effects[-1])[0,1],
            'memory_strength': np.max(self.model.memory_trace),
            'memory_width': self._calculate_peak_width(self.model.memory_trace)
        }
        
        success = metrics['memory_correlation'] > 0.5  # Should show memory effects
        
        self.results.append(TestResult(
            name="Memory Effects",
            success=success,
            metrics=metrics,
            description="Tests memory trace effects on movement preparation"
        ))
    
    def _calculate_convergence_time(self, results):
        """Calculate time to stable peak formation"""
        peak_values = np.max(results, axis=1)
        threshold = 0.9 * peak_values[-1]
        convergence_time = np.where(peak_values > threshold)[0][0]
        return convergence_time
    
    def _calculate_peak_width(self, activation):
        """Calculate width of activation peak"""
        threshold = 0.5 * np.max(activation)
        above_threshold = activation > threshold
        return np.sum(above_threshold)
    
    def visualize_results(self):
        """Create comprehensive visualization of test results"""
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Basic functionality
        plt.subplot(2, 2, 1)
        basic_test = next(r for r in self.results 
                         if r.name == "Basic Functionality")
        plt.plot(basic_test.metrics['peak_position_error'])
        plt.title("Basic Functionality Test")
        plt.ylabel("Peak Position Error")
        
        # Plot 2: Stability analysis
        plt.subplot(2, 2, 2)
        stability_test = next(r for r in self.results 
                            if r.name == "Stability Analysis")
        plt.plot(stability_test.metrics['position_std'])
        plt.title("Stability Analysis")
        plt.ylabel("Position STD")
        
        # Plot 3: Noise robustness
        plt.subplot(2, 2, 3)
        noise_test = next(r for r in self.results 
                         if r.name == "Noise Robustness")
        plt.plot(noise_test.metrics['noise_levels'], 
                noise_test.metrics['stability_measures'])
        plt.title("Noise Robustness")
        plt.xlabel("Noise Level")
        plt.ylabel("Stability Measure")
        
        # Plot 4: Adaptation effects
        plt.subplot(2, 2, 4)
        adapt_test = next(r for r in self.results 
                         if r.name == "Adaptation Analysis")
        plt.plot(adapt_test.metrics['adaptation_times'])
        plt.title("Adaptation Effects")
        plt.xlabel("Repetition")
        plt.ylabel("Preparation Time")
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """Generate detailed test report"""
        report = ["DFT Model Test Report\n====================\n"]
        
        for result in self.results:
            report.append(f"\nTest: {result.name}")
            report.append(f"Success: {'✓' if result.success else '✗'}")
            report.append(f"Description: {result.description}")
            report.append("\nMetrics:")
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    report.append(f"  {metric}: {value:.3f}")
                else:
                    report.append(f"  {metric}: {value}")
            report.append("\n" + "-"*40)
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    from enhanced_dft_model import EnhancedDFTModel
    
    # Create model and tester
    model = EnhancedDFTModel()
    tester = DFTModelTester(model)
    
    # Run tests
    results = tester.run_all_tests()
    
    # Generate visualizations and report
    tester.visualize_results()
    report = tester.generate_report()
    print(report)