/**
 * Enhanced IoT BotScan Analytics JavaScript
 * Author: Kotiwale Sumesh Singh (160124862043)
 *
 * Handles analytics functionality, data visualization, and interactive charts
 */

class AnalyticsManager {
  constructor() {
    this.charts = {};
    this.currentTab = "performance";
    this.filters = {
      dataset: "all",
      metric: "accuracy",
      timeRange: "24h",
    };

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.initializeCharts();
    this.loadAnalyticsData();
  }

  setupEventListeners() {
    // Tab switching
    document
      .querySelectorAll('#analyticsTabs button[data-bs-toggle="tab"]')
      .forEach((tab) => {
        tab.addEventListener("shown.bs.tab", (event) => {
          this.currentTab = event.target
            .getAttribute("data-bs-target")
            .substring(1);
          this.loadTabData(this.currentTab);
        });
      });

    // Filter changes
    document.getElementById("datasetFilter").addEventListener("change", (e) => {
      this.filters.dataset = e.target.value;
    });

    document.getElementById("metricFilter").addEventListener("change", (e) => {
      this.filters.metric = e.target.value;
    });

    document.getElementById("timeFilter").addEventListener("change", (e) => {
      this.filters.timeRange = e.target.value;
    });

    // Window resize handler
    window.addEventListener("resize", () => {
      this.resizeCharts();
    });
  }

  initializeCharts() {
    this.initializePerformanceCharts();
    this.initializeRobustnessCharts();
    this.initializeDriftCharts();
    this.initializeComparisonCharts();
  }

  initializePerformanceCharts() {
    // Performance Over Time Chart
    const performanceCtx = document.getElementById("performanceOverTimeChart");
    if (performanceCtx) {
      this.charts.performanceOverTime = new Chart(performanceCtx, {
        type: "line",
        data: {
          labels: this.generateTimeLabels(24),
          datasets: [
            {
              label: "Random Forest",
              data: this.generateRandomData(24, 0.85, 0.95),
              borderColor: "#e74c3c",
              backgroundColor: "rgba(231, 76, 60, 0.1)",
              borderWidth: 3,
              fill: true,
              tension: 0.4,
            },
            {
              label: "XGBoost",
              data: this.generateRandomData(24, 0.88, 0.96),
              borderColor: "#f39c12",
              backgroundColor: "rgba(243, 156, 18, 0.1)",
              borderWidth: 3,
              fill: true,
              tension: 0.4,
            },
            {
              label: "LightGBM",
              data: this.generateRandomData(24, 0.87, 0.95),
              borderColor: "#3498db",
              backgroundColor: "rgba(52, 152, 219, 0.1)",
              borderWidth: 3,
              fill: true,
              tension: 0.4,
            },
            {
              label: "Ensemble",
              data: this.generateRandomData(24, 0.92, 0.98),
              borderColor: "#27ae60",
              backgroundColor: "rgba(39, 174, 96, 0.1)",
              borderWidth: 3,
              fill: true,
              tension: 0.4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
            },
          },
          scales: {
            y: {
              beginAtZero: false,
              min: 0.8,
              max: 1.0,
              ticks: {
                callback: function (value) {
                  return (value * 100).toFixed(0) + "%";
                },
              },
            },
          },
        },
      });
    }

    // Performance Distribution Chart
    const distributionCtx = document.getElementById(
      "performanceDistributionChart"
    );
    if (distributionCtx) {
      this.charts.performanceDistribution = new Chart(distributionCtx, {
        type: "doughnut",
        data: {
          labels: [
            "Excellent (>95%)",
            "Good (90-95%)",
            "Fair (85-90%)",
            "Poor (<85%)",
          ],
          datasets: [
            {
              data: [45, 35, 15, 5],
              backgroundColor: ["#27ae60", "#3498db", "#f39c12", "#e74c3c"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
            },
          },
        },
      });
    }
  }

  initializeRobustnessCharts() {
    // Adversarial Resistance Chart
    const adversarialCtx = document.getElementById(
      "adversarialResistanceChart"
    );
    if (adversarialCtx) {
      this.charts.adversarialResistance = new Chart(adversarialCtx, {
        type: "radar",
        data: {
          labels: ["FGSM", "PGD", "C&W", "DeepFool", "AutoAttack"],
          datasets: [
            {
              label: "Random Forest",
              data: [85, 78, 72, 80, 75],
              borderColor: "#e74c3c",
              backgroundColor: "rgba(231, 76, 60, 0.2)",
              borderWidth: 2,
            },
            {
              label: "XGBoost",
              data: [88, 82, 76, 84, 79],
              borderColor: "#f39c12",
              backgroundColor: "rgba(243, 156, 18, 0.2)",
              borderWidth: 2,
            },
            {
              label: "LightGBM",
              data: [87, 81, 75, 83, 78],
              borderColor: "#3498db",
              backgroundColor: "rgba(52, 152, 219, 0.2)",
              borderWidth: 2,
            },
            {
              label: "Ensemble",
              data: [92, 87, 82, 89, 85],
              borderColor: "#27ae60",
              backgroundColor: "rgba(39, 174, 96, 0.2)",
              borderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
            },
          },
          scales: {
            r: {
              beginAtZero: true,
              max: 100,
              ticks: {
                callback: function (value) {
                  return value + "%";
                },
              },
            },
          },
        },
      });
    }

    // Robustness Comparison Chart
    const robustnessCtx = document.getElementById("robustnessComparisonChart");
    if (robustnessCtx) {
      this.charts.robustnessComparison = new Chart(robustnessCtx, {
        type: "bar",
        data: {
          labels: [
            "Noise",
            "Outliers",
            "Missing Values",
            "Feature Corruption",
            "Adversarial",
          ],
          datasets: [
            {
              label: "Random Forest",
              data: [87, 91, 89, 85, 78],
              backgroundColor: "#e74c3c",
            },
            {
              label: "XGBoost",
              data: [89, 93, 91, 87, 82],
              backgroundColor: "#f39c12",
            },
            {
              label: "LightGBM",
              data: [88, 92, 90, 86, 81],
              backgroundColor: "#3498db",
            },
            {
              label: "Ensemble",
              data: [93, 96, 94, 91, 87],
              backgroundColor: "#27ae60",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: {
                callback: function (value) {
                  return value + "%";
                },
              },
            },
          },
        },
      });
    }
  }

  initializeDriftCharts() {
    // Drift Timeline Chart
    const driftCtx = document.getElementById("driftTimelineChart");
    if (driftCtx) {
      this.charts.driftTimeline = new Chart(driftCtx, {
        type: "line",
        data: {
          labels: this.generateTimeLabels(48),
          datasets: [
            {
              label: "Performance",
              data: this.generateDriftData(48),
              borderColor: "#27ae60",
              backgroundColor: "rgba(39, 174, 96, 0.1)",
              borderWidth: 3,
              fill: true,
              tension: 0.4,
            },
            {
              label: "Drift Detection",
              data: this.generateDriftDetectionData(48),
              borderColor: "#e74c3c",
              backgroundColor: "rgba(231, 76, 60, 0.1)",
              borderWidth: 2,
              fill: false,
              tension: 0.4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 1.0,
              ticks: {
                callback: function (value) {
                  return (value * 100).toFixed(0) + "%";
                },
              },
            },
          },
        },
      });
    }

    // Drift Types Chart
    const driftTypesCtx = document.getElementById("driftTypesChart");
    if (driftTypesCtx) {
      this.charts.driftTypes = new Chart(driftTypesCtx, {
        type: "pie",
        data: {
          labels: [
            "Covariate Shift",
            "Concept Shift",
            "Prior Shift",
            "Label Shift",
          ],
          datasets: [
            {
              data: [40, 30, 20, 10],
              backgroundColor: ["#e74c3c", "#f39c12", "#3498db", "#27ae60"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
            },
          },
        },
      });
    }
  }

  initializeComparisonCharts() {
    // Model Comparison Chart
    const comparisonCtx = document.getElementById("modelComparisonChart");
    if (comparisonCtx) {
      this.charts.modelComparison = new Chart(comparisonCtx, {
        type: "bar",
        data: {
          labels: ["Random Forest", "XGBoost", "LightGBM", "Ensemble"],
          datasets: [
            {
              label: "Accuracy",
              data: [0.92, 0.94, 0.93, 0.96],
              backgroundColor: "#27ae60",
            },
            {
              label: "Precision",
              data: [0.89, 0.91, 0.9, 0.94],
              backgroundColor: "#3498db",
            },
            {
              label: "Recall",
              data: [0.87, 0.89, 0.88, 0.92],
              backgroundColor: "#f39c12",
            },
            {
              label: "F1-Score",
              data: [0.88, 0.9, 0.89, 0.93],
              backgroundColor: "#9b59b6",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 1.0,
              ticks: {
                callback: function (value) {
                  return (value * 100).toFixed(0) + "%";
                },
              },
            },
          },
        },
      });
    }

    // Performance Trends Chart
    const trendsCtx = document.getElementById("performanceTrendsChart");
    if (trendsCtx) {
      this.charts.performanceTrends = new Chart(trendsCtx, {
        type: "line",
        data: {
          labels: this.generateTimeLabels(7),
          datasets: [
            {
              label: "Random Forest",
              data: this.generateRandomData(7, 0.9, 0.94),
              borderColor: "#e74c3c",
              backgroundColor: "rgba(231, 76, 60, 0.1)",
              borderWidth: 3,
              fill: false,
              tension: 0.4,
            },
            {
              label: "XGBoost",
              data: this.generateRandomData(7, 0.92, 0.96),
              borderColor: "#f39c12",
              backgroundColor: "rgba(243, 156, 18, 0.1)",
              borderWidth: 3,
              fill: false,
              tension: 0.4,
            },
            {
              label: "LightGBM",
              data: this.generateRandomData(7, 0.91, 0.95),
              borderColor: "#3498db",
              backgroundColor: "rgba(52, 152, 219, 0.1)",
              borderWidth: 3,
              fill: false,
              tension: 0.4,
            },
            {
              label: "Ensemble",
              data: this.generateRandomData(7, 0.94, 0.98),
              borderColor: "#27ae60",
              backgroundColor: "rgba(39, 174, 96, 0.1)",
              borderWidth: 3,
              fill: false,
              tension: 0.4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
            },
          },
          scales: {
            y: {
              beginAtZero: false,
              min: 0.85,
              max: 1.0,
              ticks: {
                callback: function (value) {
                  return (value * 100).toFixed(0) + "%";
                },
              },
            },
          },
        },
      });
    }
  }

  generateTimeLabels(hours) {
    const labels = [];
    const now = new Date();

    for (let i = hours - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000);
      labels.push(
        time.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    }

    return labels;
  }

  generateRandomData(count, min, max) {
    const data = [];
    for (let i = 0; i < count; i++) {
      data.push(Math.random() * (max - min) + min);
    }
    return data;
  }

  generateDriftData(count) {
    const data = [];
    let baseValue = 0.95;

    for (let i = 0; i < count; i++) {
      // Simulate drift events
      if (i === 12 || i === 24 || i === 36) {
        baseValue -= 0.1; // Drift event
      } else if (i === 18 || i === 30 || i === 42) {
        baseValue += 0.05; // Recovery
      }

      data.push(baseValue + (Math.random() - 0.5) * 0.02);
    }

    return data;
  }

  generateDriftDetectionData(count) {
    const data = [];

    for (let i = 0; i < count; i++) {
      // Drift detection spikes
      if (i === 12 || i === 24 || i === 36) {
        data.push(0.8);
      } else {
        data.push(Math.random() * 0.1);
      }
    }

    return data;
  }

  loadAnalyticsData() {
    this.loadPerformanceData();
    this.loadRobustnessData();
    this.loadDriftData();
    this.loadComparisonData();
  }

  loadTabData(tabName) {
    switch (tabName) {
      case "performance":
        this.loadPerformanceData();
        break;
      case "robustness":
        this.loadRobustnessData();
        break;
      case "drift":
        this.loadDriftData();
        break;
      case "comparison":
        this.loadComparisonData();
        break;
    }
  }

  loadPerformanceData() {
    this.updatePerformanceTable();
  }

  loadRobustnessData() {
    // Update robustness metrics
    this.updateRobustnessMetrics();
  }

  loadDriftData() {
    // Update drift metrics
    this.updateDriftMetrics();
  }

  loadComparisonData() {
    this.updateComparisonTable();
  }

  updatePerformanceTable() {
    const tableBody = document.getElementById("performanceTable");
    if (!tableBody) return;

    const performanceData = [
      {
        model: "Random Forest",
        dataset: "N-BaIoT",
        accuracy: 0.92,
        precision: 0.89,
        recall: 0.87,
        f1: 0.88,
        roc: 0.94,
        time: "2.3s",
        status: "Active",
      },
      {
        model: "XGBoost",
        dataset: "N-BaIoT",
        accuracy: 0.94,
        precision: 0.91,
        recall: 0.89,
        f1: 0.9,
        roc: 0.96,
        time: "1.8s",
        status: "Active",
      },
      {
        model: "LightGBM",
        dataset: "N-BaIoT",
        accuracy: 0.93,
        precision: 0.9,
        recall: 0.88,
        f1: 0.89,
        roc: 0.95,
        time: "1.5s",
        status: "Active",
      },
      {
        model: "Ensemble",
        dataset: "N-BaIoT",
        accuracy: 0.96,
        precision: 0.94,
        recall: 0.92,
        f1: 0.93,
        roc: 0.98,
        time: "3.2s",
        status: "Active",
      },
      {
        model: "Random Forest",
        dataset: "IoT-23",
        accuracy: 0.89,
        precision: 0.86,
        recall: 0.84,
        f1: 0.85,
        roc: 0.91,
        time: "2.1s",
        status: "Active",
      },
      {
        model: "XGBoost",
        dataset: "IoT-23",
        accuracy: 0.91,
        precision: 0.88,
        recall: 0.86,
        f1: 0.87,
        roc: 0.93,
        time: "1.6s",
        status: "Active",
      },
      {
        model: "LightGBM",
        dataset: "IoT-23",
        accuracy: 0.9,
        precision: 0.87,
        recall: 0.85,
        f1: 0.86,
        roc: 0.92,
        time: "1.3s",
        status: "Active",
      },
      {
        model: "Ensemble",
        dataset: "IoT-23",
        accuracy: 0.93,
        precision: 0.9,
        recall: 0.88,
        f1: 0.89,
        roc: 0.95,
        time: "2.8s",
        status: "Active",
      },
    ];

    tableBody.innerHTML = performanceData
      .map(
        (row) => `
            <tr>
                <td><strong>${row.model}</strong></td>
                <td><span class="badge bg-info">${row.dataset}</span></td>
                <td><span class="badge bg-success">${(
                  row.accuracy * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-primary">${(
                  row.precision * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-warning">${(row.recall * 100).toFixed(
                  1
                )}%</span></td>
                <td><span class="badge bg-secondary">${(row.f1 * 100).toFixed(
                  1
                )}%</span></td>
                <td><span class="badge bg-dark">${(row.roc * 100).toFixed(
                  1
                )}%</span></td>
                <td><small>${row.time}</small></td>
                <td><span class="badge bg-success">${row.status}</span></td>
            </tr>
        `
      )
      .join("");
  }

  updateRobustnessMetrics() {
    // Update robustness metric cards
    const metrics = {
      "noise-robustness": Math.floor(Math.random() * 10 + 85),
      "adversarial-robustness": Math.floor(Math.random() * 10 + 80),
      "outlier-robustness": Math.floor(Math.random() * 10 + 88),
    };

    Object.entries(metrics).forEach(([id, value]) => {
      const element = document.getElementById(id);
      if (element) {
        element.textContent = value + "%";

        // Update progress bar
        const progressBar =
          element.parentElement.querySelector(".progress-bar");
        if (progressBar) {
          progressBar.style.width = value + "%";

          // Update color based on value
          progressBar.className = "progress-bar";
          if (value > 90) {
            progressBar.classList.add("bg-success");
          } else if (value > 80) {
            progressBar.classList.add("bg-warning");
          } else {
            progressBar.classList.add("bg-danger");
          }
        }
      }
    });
  }

  updateDriftMetrics() {
    // Update drift metrics
    const driftDetections = Math.floor(Math.random() * 5 + 1);
    document.getElementById("drift-detections").textContent = driftDetections;

    const adaptationTime = (Math.random() * 2 + 1).toFixed(1);
    document.getElementById("adaptation-time").textContent =
      adaptationTime + "s";

    const driftAccuracy = (Math.random() * 5 + 92).toFixed(1);
    document.getElementById("drift-accuracy").textContent = driftAccuracy + "%";
  }

  updateComparisonTable() {
    const tableBody = document.getElementById("comparisonTable");
    if (!tableBody) return;

    const comparisonData = [
      {
        model: "Ensemble",
        accuracy: 0.96,
        precision: 0.94,
        recall: 0.92,
        f1: 0.93,
        robustness: 0.89,
        time: "3.2s",
        rank: 1,
      },
      {
        model: "XGBoost",
        accuracy: 0.94,
        precision: 0.91,
        recall: 0.89,
        f1: 0.9,
        robustness: 0.85,
        time: "1.8s",
        rank: 2,
      },
      {
        model: "LightGBM",
        accuracy: 0.93,
        precision: 0.9,
        recall: 0.88,
        f1: 0.89,
        robustness: 0.84,
        time: "1.5s",
        rank: 3,
      },
      {
        model: "Random Forest",
        accuracy: 0.92,
        precision: 0.89,
        recall: 0.87,
        f1: 0.88,
        robustness: 0.82,
        time: "2.3s",
        rank: 4,
      },
    ];

    tableBody.innerHTML = comparisonData
      .map(
        (row) => `
            <tr>
                <td><strong>${row.model}</strong></td>
                <td><span class="badge bg-success">${(
                  row.accuracy * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-primary">${(
                  row.precision * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-warning">${(row.recall * 100).toFixed(
                  1
                )}%</span></td>
                <td><span class="badge bg-secondary">${(row.f1 * 100).toFixed(
                  1
                )}%</span></td>
                <td><span class="badge bg-info">${(
                  row.robustness * 100
                ).toFixed(1)}%</span></td>
                <td><small>${row.time}</small></td>
                <td><span class="badge bg-${
                  row.rank === 1
                    ? "success"
                    : row.rank === 2
                    ? "warning"
                    : row.rank === 3
                    ? "info"
                    : "secondary"
                }">#${row.rank}</span></td>
            </tr>
        `
      )
      .join("");
  }

  applyFilters() {
    console.log("Applying filters:", this.filters);

    // Update charts based on filters
    this.updateChartsWithFilters();

    // Update tables based on filters
    this.updateTablesWithFilters();
  }

  updateChartsWithFilters() {
    // Update charts based on current filters
    // This would typically involve API calls to get filtered data
    console.log("Updating charts with filters");
  }

  updateTablesWithFilters() {
    // Update tables based on current filters
    console.log("Updating tables with filters");
  }

  resizeCharts() {
    Object.values(this.charts).forEach((chart) => {
      if (chart) {
        chart.resize();
      }
    });
  }

  destroy() {
    Object.values(this.charts).forEach((chart) => {
      if (chart) {
        chart.destroy();
      }
    });
  }
}

// Global functions for button clicks
function refreshAnalytics() {
  analyticsManager.loadAnalyticsData();
}

function exportAnalytics() {
  console.log("Exporting analytics data...");
  // This would typically generate a CSV or PDF report
}

function applyFilters() {
  analyticsManager.applyFilters();
}

// Initialize analytics manager when DOM is loaded
let analyticsManager;

document.addEventListener("DOMContentLoaded", () => {
  analyticsManager = new AnalyticsManager();
});

// Cleanup on page unload
window.addEventListener("beforeunload", () => {
  if (analyticsManager) {
    analyticsManager.destroy();
  }
});
