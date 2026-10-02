/**
 * High-Performance Multithreaded Monte Carlo Peak Forward Exposure (PFE) Engine in C++17/C++20.
 * Exports both an extern "C" ABI for direct Python ctypes binding and a standalone benchmark CLI.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <numeric>

extern "C" {
    struct C_PFEPoint {
        double tenor;
        double expected_exposure;
        double pfe_95;
        double pfe_97_5;
        double pfe_99;
    };
}

class MonteCarloPFESimulator {
public:
    MonteCarloPFESimulator(
        int num_simulations = 100000,
        double mean_reversion = 0.03,
        double volatility = 0.15,
        double initial_mtm = 1250000.0,
        double notional = 50000000.0
    ) : num_simulations_(num_simulations),
        mean_reversion_(mean_reversion),
        volatility_(volatility),
        initial_mtm_(initial_mtm),
        notional_(notional) {}

    std::vector<C_PFEPoint> run_simulation(const std::vector<double>& tenors, int num_threads = 0) {
        if (num_threads <= 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }

        const size_t num_steps = tenors.size();
        std::vector<std::vector<double>> global_exposures(num_steps, std::vector<double>(num_simulations_));

        int paths_per_thread = num_simulations_ / num_threads;
        std::vector<std::thread> workers;

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            int start_path = t_idx * paths_per_thread;
            int end_path = (t_idx == num_threads - 1) ? num_simulations_ : start_path + paths_per_thread;

            workers.emplace_back([this, &tenors, &global_exposures, start_path, end_path, t_idx]() {
                std::mt19937_64 rng(1337 + t_idx * 997);
                std::normal_distribution<double> norm_dist(0.0, 1.0);

                for (int p = start_path; p < end_path; ++p) {
                    double current_x = 0.0;
                    double prev_t = 0.0;

                    for (size_t step = 0; step < tenors.size(); ++step) {
                        double t = tenors[step];
                        double dt = t - prev_t;
                        prev_t = t;

                        double drift = current_x * std::exp(-mean_reversion_ * dt);
                        double var = (volatility_ * volatility_ / (2.0 * mean_reversion_)) * (1.0 - std::exp(-2.0 * mean_reversion_ * dt));
                        double diffusion = std::sqrt(std::max(0.0, var)) * norm_dist(rng);

                        current_x = drift + diffusion;

                        double maturity = 10.0;
                        double active_weight = std::max(0.0, (maturity - t) / maturity);
                        double mtm = initial_mtm_ * active_weight + current_x * notional_ * active_weight * 0.1;
                        global_exposures[step][p] = std::max(0.0, mtm);
                    }
                }
            });
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }

        std::vector<C_PFEPoint> results;
        results.reserve(num_steps);

        for (size_t step = 0; step < num_steps; ++step) {
            auto& row = global_exposures[step];
            std::sort(row.begin(), row.end());

            double sum = std::accumulate(row.begin(), row.end(), 0.0);
            double ee = sum / num_simulations_;

            size_t idx_95 = static_cast<size_t>(num_simulations_ * 0.95);
            size_t idx_97_5 = static_cast<size_t>(num_simulations_ * 0.975);
            size_t idx_99 = static_cast<size_t>(num_simulations_ * 0.99);

            results.push_back({
                tenors[step],
                ee,
                row[idx_95],
                row[idx_97_5],
                row[idx_99]
            });
        }

        return results;
    }

private:
    int num_simulations_;
    double mean_reversion_;
    double volatility_;
    double initial_mtm_;
    double notional_;
};

// C-ABI Exported Function for Python ctypes
extern "C" {
    int run_monte_carlo_pfe_c(
        int num_simulations,
        double mean_reversion,
        double volatility,
        double initial_mtm,
        double notional,
        const double* tenors,
        int num_tenors,
        int num_threads,
        C_PFEPoint* out_results
    ) {
        if (!tenors || !out_results || num_tenors <= 0 || num_simulations <= 0) {
            return -1;
        }

        std::vector<double> tenors_vec(tenors, tenors + num_tenors);
        MonteCarloPFESimulator simulator(num_simulations, mean_reversion, volatility, initial_mtm, notional);
        auto res = simulator.run_simulation(tenors_vec, num_threads);

        for (int i = 0; i < num_tenors; ++i) {
            out_results[i] = res[i];
        }

        return 0;
    }
}

#ifndef BUILD_SHARED_LIB
int main(int argc, char* argv[]) {
    std::cout << "=================================================================\n";
    std::cout << " High-Performance C++20 Multithreaded Monte Carlo PFE Risk Engine\n";
    std::cout << "=================================================================\n";

    int num_paths = 100000;
    if (argc > 1) {
        num_paths = std::stoi(argv[1]);
    }

    std::vector<double> tenors = {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0};
    unsigned int hardware_threads = std::thread::hardware_concurrency();

    std::cout << "Config: " << num_paths << " paths | " << tenors.size() << " tenors | "
              << hardware_threads << " worker threads" << std::endl;

    MonteCarloPFESimulator simulator(num_paths);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = simulator.run_simulation(tenors, hardware_threads);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = (num_paths * tenors.size()) / (elapsed_ms / 1000.0);

    std::cout << "\nResults Summary:\n";
    std::cout << std::string(75, '-') << "\n";
    std::cout << std::setw(12) << "Tenor (Yrs)"
              << std::setw(16) << "EE (USD)"
              << std::setw(16) << "PFE 95% (USD)"
              << std::setw(16) << "PFE 97.5% (USD)"
              << std::setw(15) << "PFE 99% (USD)" << "\n";
    std::cout << std::string(75, '-') << "\n";

    double peak_pfe = 0.0;
    double peak_tenor = 0.0;

    for (const auto& r : results) {
        if (r.pfe_97_5 > peak_pfe) {
            peak_pfe = r.pfe_97_5;
            peak_tenor = r.tenor;
        }
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << r.tenor << "y"
                  << std::setw(16) << r.expected_exposure
                  << std::setw(16) << r.pfe_95
                  << std::setw(16) << r.pfe_97_5
                  << std::setw(15) << r.pfe_99 << "\n";
    }
    std::cout << std::string(75, '-') << "\n";

    std::cout << "\n[Performance Benchmark]\n";
    std::cout << " * Computation Time: " << elapsed_ms << " ms\n";
    std::cout << " * Simulation Throughput: " << std::fixed << std::setprecision(0) << throughput << " valuation events/sec\n";
    std::cout << " * Peak Forward Exposure (97.5%): $" << std::fixed << std::setprecision(2) << peak_pfe
              << " at " << peak_tenor << " years\n";

    return 0;
}
#endif
