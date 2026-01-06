"""
ARM Visualization Module
Generates charts and visualizations for robustness analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ARMVisualizer:
    """Generate visualizations for ARM robustness results."""
    
    def __init__(self):
        self.color_scheme = {
            'noise': '#FF6B6B',
            'masking': '#4ECDC4', 
            'burst': '#45B7D1',
            'overall': '#96CEB4',
            'baseline': '#6C757D',
            'improved': '#28A745'
        }
    
    def create_robustness_comparison(self, before: Dict[str, float], 
                                     after: Dict[str, float]) -> go.Figure:
        """Create before/after robustness comparison bar chart."""
        
        categories = ['Overall', 'Noise', 'Masking', 'Burst', 'Confidence']
        before_vals = [
            before.get('overall_robustness', 0),
            before.get('noise_robustness', 0),
            before.get('masking_robustness', 0),
            before.get('burst_robustness', 0),
            before.get('confidence_stability', 0)
        ]
        after_vals = [
            after.get('overall_robustness', 0),
            after.get('noise_robustness', 0),
            after.get('masking_robustness', 0),
            after.get('burst_robustness', 0),
            after.get('confidence_stability', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(name='Before ARM', x=categories, y=before_vals, 
                   marker_color=self.color_scheme['baseline']),
            go.Bar(name='After ARM', x=categories, y=after_vals,
                   marker_color=self.color_scheme['improved'])
        ])
        
        fig.update_layout(
            title='Robustness Improvement with ARM Training',
            xaxis_title='Robustness Category',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            barmode='group',
            template='plotly_dark',
            legend=dict(x=0.7, y=1.1, orientation='h')
        )
        
        return fig
    
    def create_threat_heatmap(self, results: Dict[str, Any]) -> go.Figure:
        """Create heatmap showing robustness across threat scenarios."""
        
        scenarios = results.get('threat_scenarios', {})
        
        # Build data matrix
        threat_types = []
        scenario_names = []
        robustness_values = []
        
        for threat_type, threat_results in scenarios.items():
            for scenario, metrics in threat_results.items():
                threat_types.append(threat_type.title())
                scenario_names.append(scenario)
                robustness_values.append(metrics.get('robustness_score', 0))
        
        # Create pivot-style data
        df = pd.DataFrame({
            'Threat': threat_types,
            'Scenario': scenario_names,
            'Robustness': robustness_values
        })
        
        fig = px.scatter(df, x='Scenario', y='Threat', size='Robustness',
                        color='Robustness', color_continuous_scale='RdYlGn',
                        title='Robustness by Threat Scenario',
                        size_max=40)
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_tickangle=45
        )
        
        return fig
    
    def create_scenario_bar_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create bar chart of robustness by scenario."""
        
        scenarios = results.get('threat_scenarios', {})
        
        data = []
        for threat_type, threat_results in scenarios.items():
            for scenario, metrics in threat_results.items():
                data.append({
                    'Threat': threat_type.title(),
                    'Scenario': scenario,
                    'Robustness': metrics.get('robustness_score', 0),
                    'Accuracy': metrics.get('accuracy', 0)
                })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(df, x='Scenario', y='Robustness', color='Threat',
                    title='Robustness Score by Threat Scenario',
                    barmode='group',
                    color_discrete_map={
                        'Noise': self.color_scheme['noise'],
                        'Masking': self.color_scheme['masking'],
                        'Burst': self.color_scheme['burst']
                    })
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_tickangle=45,
            yaxis_range=[0, 1.1]
        )
        
        return fig
    
    def create_accuracy_impact_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create chart showing accuracy drop under different threats."""
        
        scenarios = results.get('threat_scenarios', {})
        
        data = []
        for threat_type, threat_results in scenarios.items():
            for scenario, metrics in threat_results.items():
                data.append({
                    'Scenario': scenario,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Accuracy Drop': metrics.get('accuracy_drop', 0),
                    'Threat': threat_type.title()
                })
        
        df = pd.DataFrame(data)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Accuracy Under Threat', 'Accuracy Drop'))
        
        for i, threat in enumerate(['Noise', 'Masking', 'Burst']):
            threat_df = df[df['Threat'] == threat]
            color = list(self.color_scheme.values())[i]
            
            fig.add_trace(
                go.Scatter(x=threat_df['Scenario'], y=threat_df['Accuracy'],
                          name=threat, mode='lines+markers',
                          line=dict(color=color)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=threat_df['Scenario'], y=threat_df['Accuracy Drop'],
                      name=f'{threat} Drop', marker_color=color, showlegend=False),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Accuracy Impact Analysis',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_summary_gauge(self, overall_score: float) -> go.Figure:
        """Create gauge chart for overall robustness score."""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score * 100,
            title={'text': "Overall Robustness Score"},
            delta={'reference': 83.08, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': '%'},
                'bar': {'color': self.color_scheme['overall']},
                'steps': [
                    {'range': [0, 80], 'color': '#FF6B6B'},
                    {'range': [80, 90], 'color': '#FFE66D'},
                    {'range': [90, 95], 'color': '#4ECDC4'},
                    {'range': [95, 100], 'color': '#45B7D1'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=300
        )
        
        return fig


def add_visualizations_to_streamlit(arm_results: Dict[str, Any]):
    """Helper function to add ARM visualizations to Streamlit."""
    import streamlit as st
    
    viz = ARMVisualizer()
    
    # Overall gauge
    overall_score = arm_results.get('aggregate_scores', {}).get('overall_robustness', 0)
    st.plotly_chart(viz.create_summary_gauge(overall_score), use_container_width=True)
    
    # Scenario bar chart
    st.plotly_chart(viz.create_scenario_bar_chart(arm_results), use_container_width=True)
    
    # Accuracy impact
    st.plotly_chart(viz.create_accuracy_impact_chart(arm_results), use_container_width=True)
