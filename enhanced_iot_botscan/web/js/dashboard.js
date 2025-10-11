/**
 * Enhanced IoT BotScan Dashboard JavaScript
 * Author: Kotiwale Sumesh Singh (160124862043)
 *
 * Handles dashboard functionality, real-time updates, and data visualization
 */

class DashboardManager {
  constructor() {
    this.charts = {};
    this.websocket = null;
    this.refreshInterval = null;
    this.isConnected = false;

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.initializeCharts();
    this.startRealTimeUpdates();
    this.loadInitialData();
  }

  setupEventListeners() {
    // Sidebar navigation
    document.querySelectorAll(".sidebar .nav-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        this.switchSection(link.dataset.section);
      });
    });

    // Mobile menu toggle
    const navbarToggler = document.querySelector(".navbar-toggler");
    if (navbarToggler) {
      navbarToggler.addEventListener("click", () => {
        document.querySelector(".sidebar").classList.toggle("show");
      });
    }

    // Window resize handler
    window.addEventListener("resize", () => {
      this.resizeCharts();
    });
  }

  switchSection(sectionName) {
    // Hide all sections
    document.querySelectorAll(".content-section").forEach((section) => {
      section.style.display = "none";
    });

    // Show selected section
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
      targetSection.style.display = "block";
    }

    // Update active nav link
    document.querySelectorAll(".sidebar .nav-link").forEach((link) => {
      link.classList.remove("active");
    });
    document
      .querySelector(`[data-section="${sectionName}"]`)
      .classList.add("active");

    // Load section-specific data
    this.loadSectionData(sectionName);
  }

  initializeCharts() {
    this.initializeThreatsChart();
    this.initializeThreatDistributionChart();
    this.initializePerformanceChart();
  }

  initializeThreatsChart() {
    const ctx = document.getElementById("threatsChart");
    if (!ctx) return;

    this.charts.threats = new Chart(ctx, {
      type: "line",
      data: {
        labels: this.generateTimeLabels(24),
        datasets: [
          {
            label: "Threats Detected",
            data: this.generateRandomData(24, 0, 100),
            borderColor: "#e74c3c",
            backgroundColor: "rgba(231, 76, 60, 0.1)",
            borderWidth: 3,
            fill: true,
            tension: 0.4,
          },
          {
            label: "False Positives",
            data: this.generateRandomData(24, 0, 20),
            borderColor: "#f39c12",
            backgroundColor: "rgba(243, 156, 18, 0.1)",
            borderWidth: 2,
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
          title: {
            display: false,
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: {
              color: "rgba(0,0,0,0.1)",
            },
          },
          x: {
            grid: {
              color: "rgba(0,0,0,0.1)",
            },
          },
        },
        interaction: {
          intersect: false,
          mode: "index",
        },
      },
    });
  }

  initializeThreatDistributionChart() {
    const ctx = document.getElementById("threatDistributionChart");
    if (!ctx) return;

    this.charts.threatDistribution = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Botnet", "Malware", "DDoS", "Other"],
        datasets: [
          {
            data: [45, 25, 20, 10],
            backgroundColor: ["#e74c3c", "#f39c12", "#3498db", "#95a5a6"],
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

  initializePerformanceChart() {
    const ctx = document.getElementById("performanceChart");
    if (!ctx) return;

    this.charts.performance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Random Forest", "XGBoost", "LightGBM", "Ensemble"],
        datasets: [
          {
            label: "Accuracy",
            data: [0.92, 0.94, 0.93, 0.96],
            backgroundColor: "#27ae60",
            borderColor: "#27ae60",
            borderWidth: 1,
          },
          {
            label: "Precision",
            data: [0.89, 0.91, 0.9, 0.94],
            backgroundColor: "#3498db",
            borderColor: "#3498db",
            borderWidth: 1,
          },
          {
            label: "Recall",
            data: [0.87, 0.89, 0.88, 0.92],
            backgroundColor: "#f39c12",
            borderColor: "#f39c12",
            borderWidth: 1,
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
            max: 1,
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
      data.push(Math.floor(Math.random() * (max - min + 1)) + min);
    }
    return data;
  }

  loadInitialData() {
    this.showLoading(true);

    // Simulate API call
    setTimeout(() => {
      this.updateDashboardData();
      this.updateModelPerformanceTable();
      this.updateRecentAlerts();
      this.showLoading(false);
    }, 1000);
  }

  updateDashboardData() {
    // Update key metrics
    document.getElementById("total-threats").textContent =
      Math.floor(Math.random() * 1000) + 500;
    document.getElementById("active-models").textContent = "4";
    document.getElementById("accuracy-rate").textContent =
      (Math.random() * 10 + 90).toFixed(1) + "%";
    document.getElementById("drift-alerts").textContent = Math.floor(
      Math.random() * 5
    );

    // Update system metrics
    this.updateSystemMetrics();
  }

  updateSystemMetrics() {
    const metrics = {
      "cpu-usage": Math.floor(Math.random() * 30 + 30),
      "memory-usage": Math.floor(Math.random() * 20 + 60),
      "disk-usage": Math.floor(Math.random() * 10 + 20),
      "network-load": Math.floor(Math.random() * 15 + 5),
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
          if (value > 80) {
            progressBar.classList.add("bg-danger");
          } else if (value > 60) {
            progressBar.classList.add("bg-warning");
          } else {
            progressBar.classList.add("bg-success");
          }
        }
      }
    });
  }

  updateModelPerformanceTable() {
    const tableBody = document.getElementById("modelPerformanceTable");
    if (!tableBody) return;

    const models = [
      {
        name: "Random Forest",
        accuracy: 0.92,
        precision: 0.89,
        recall: 0.87,
        status: "Active",
      },
      {
        name: "XGBoost",
        accuracy: 0.94,
        precision: 0.91,
        recall: 0.89,
        status: "Active",
      },
      {
        name: "LightGBM",
        accuracy: 0.93,
        precision: 0.9,
        recall: 0.88,
        status: "Active",
      },
      {
        name: "Ensemble",
        accuracy: 0.96,
        precision: 0.94,
        recall: 0.92,
        status: "Active",
      },
    ];

    tableBody.innerHTML = models
      .map(
        (model) => `
            <tr>
                <td><strong>${model.name}</strong></td>
                <td><span class="badge bg-success">${(
                  model.accuracy * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-info">${(
                  model.precision * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-warning">${(
                  model.recall * 100
                ).toFixed(1)}%</span></td>
                <td><span class="badge bg-success">${model.status}</span></td>
            </tr>
        `
      )
      .join("");
  }

  updateRecentAlerts() {
    const alertsContainer = document.getElementById("recentAlerts");
    if (!alertsContainer) return;

    const alerts = [
      {
        type: "warning",
        message: "High false positive rate detected in Random Forest model",
        time: "2 minutes ago",
      },
      {
        type: "info",
        message: "New dataset loaded: IoT-23 validation set",
        time: "15 minutes ago",
      },
      {
        type: "success",
        message: "Model retraining completed successfully",
        time: "1 hour ago",
      },
      {
        type: "danger",
        message: "Concept drift detected in network traffic patterns",
        time: "2 hours ago",
      },
    ];

    alertsContainer.innerHTML = alerts
      .map(
        (alert) => `
            <div class="alert alert-${
              alert.type
            } alert-dismissible fade show" role="alert">
                <i class="fas fa-${this.getAlertIcon(alert.type)} me-2"></i>
                ${alert.message}
                <small class="text-muted d-block mt-1">${alert.time}</small>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `
      )
      .join("");
  }

  getAlertIcon(type) {
    const icons = {
      success: "check-circle",
      info: "info-circle",
      warning: "exclamation-triangle",
      danger: "exclamation-circle",
    };
    return icons[type] || "info-circle";
  }

  loadSectionData(sectionName) {
    switch (sectionName) {
      case "analytics":
        this.loadAnalyticsData();
        break;
      case "models":
        this.loadModelsData();
        break;
      case "datasets":
        this.loadDatasetsData();
        break;
      case "adversarial":
        this.loadAdversarialData();
        break;
      case "drift":
        this.loadDriftData();
        break;
      case "settings":
        this.loadSettingsData();
        break;
    }
  }

  loadAnalyticsData() {
    // Load analytics-specific data
    console.log("Loading analytics data...");
  }

  loadModelsData() {
    // Load models-specific data
    console.log("Loading models data...");
  }

  loadDatasetsData() {
    // Load datasets-specific data
    console.log("Loading datasets data...");
  }

  loadAdversarialData() {
    // Load adversarial testing data
    console.log("Loading adversarial data...");
  }

  loadDriftData() {
    // Load concept drift data
    console.log("Loading drift data...");
  }

  loadSettingsData() {
    // Load settings data
    console.log("Loading settings data...");
  }

  startRealTimeUpdates() {
    // Update data every 30 seconds
    this.refreshInterval = setInterval(() => {
      this.updateDashboardData();
      this.updateCharts();
    }, 30000);

    // Initialize WebSocket connection
    this.initializeWebSocket();
  }

  initializeWebSocket() {
    try {
      this.websocket = new WebSocket("ws://localhost:8000/ws");

      this.websocket.onopen = () => {
        this.isConnected = true;
        console.log("WebSocket connected");
      };

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleRealTimeData(data);
      };

      this.websocket.onclose = () => {
        this.isConnected = false;
        console.log("WebSocket disconnected");
        // Attempt to reconnect after 5 seconds
        setTimeout(() => this.initializeWebSocket(), 5000);
      };

      this.websocket.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    } catch (error) {
      console.error("Failed to initialize WebSocket:", error);
    }
  }

  handleRealTimeData(data) {
    switch (data.type) {
      case "threat_detected":
        this.handleThreatDetection(data);
        break;
      case "model_update":
        this.handleModelUpdate(data);
        break;
      case "drift_alert":
        this.handleDriftAlert(data);
        break;
      case "system_metrics":
        this.handleSystemMetrics(data);
        break;
    }
  }

  handleThreatDetection(data) {
    // Update threat counter
    const currentThreats = parseInt(
      document.getElementById("total-threats").textContent
    );
    document.getElementById("total-threats").textContent = currentThreats + 1;

    // Add to recent alerts
    this.addRecentAlert(
      "warning",
      `New threat detected: ${data.threat_type}`,
      "Just now"
    );
  }

  handleModelUpdate(data) {
    // Update model performance
    this.updateModelPerformanceTable();
  }

  handleDriftAlert(data) {
    // Update drift counter
    const currentDrift = parseInt(
      document.getElementById("drift-alerts").textContent
    );
    document.getElementById("drift-alerts").textContent = currentDrift + 1;

    // Add to recent alerts
    this.addRecentAlert(
      "danger",
      `Concept drift detected: ${data.drift_type}`,
      "Just now"
    );
  }

  handleSystemMetrics(data) {
    // Update system metrics
    Object.entries(data.metrics).forEach(([key, value]) => {
      const element = document.getElementById(key);
      if (element) {
        element.textContent = value + "%";
      }
    });
  }

  addRecentAlert(type, message, time) {
    const alertsContainer = document.getElementById("recentAlerts");
    if (!alertsContainer) return;

    const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
                ${message}
                <small class="text-muted d-block mt-1">${time}</small>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

    alertsContainer.insertAdjacentHTML("afterbegin", alertHtml);
  }

  updateCharts() {
    // Update threats chart with new data
    if (this.charts.threats) {
      const newData = this.generateRandomData(1, 0, 100)[0];
      this.charts.threats.data.datasets[0].data.push(newData);
      this.charts.threats.data.datasets[0].data.shift();
      this.charts.threats.update("none");
    }

    // Update other charts as needed
  }

  resizeCharts() {
    Object.values(this.charts).forEach((chart) => {
      if (chart) {
        chart.resize();
      }
    });
  }

  showLoading(show) {
    const spinner = document.getElementById("loadingSpinner");
    if (spinner) {
      spinner.style.display = show ? "block" : "none";
    }
  }

  destroy() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }

    if (this.websocket) {
      this.websocket.close();
    }

    Object.values(this.charts).forEach((chart) => {
      if (chart) {
        chart.destroy();
      }
    });
  }
}

// Global functions for button clicks
function refreshDashboard() {
  dashboardManager.loadInitialData();
}

function exportReport() {
  // Implement report export functionality
  console.log("Exporting report...");
  // This would typically generate a PDF or CSV report
}

function refreshAnalytics() {
  dashboardManager.loadAnalyticsData();
}

function exportAnalytics() {
  // Implement analytics export functionality
  console.log("Exporting analytics data...");
}

// Initialize dashboard when DOM is loaded
let dashboardManager;

document.addEventListener("DOMContentLoaded", () => {
  dashboardManager = new DashboardManager();
});

// Cleanup on page unload
window.addEventListener("beforeunload", () => {
  if (dashboardManager) {
    dashboardManager.destroy();
  }
});
