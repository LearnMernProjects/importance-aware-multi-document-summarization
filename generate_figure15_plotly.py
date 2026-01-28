import plotly.graph_objects as go
import plotly.io as pio

# Data
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression']
baseline = [0.47, 0.30, 0.44, 0.89, 0.93, 0.38]
aims = [0.52, 0.35, 0.50, 0.92, 0.95, 0.36]

# ============================================================================
# CREATE INTERACTIVE PLOTLY FIGURE
# ============================================================================

fig = go.Figure(data=[
    go.Bar(
        name='Baseline',
        x=metrics,
        y=baseline,
        marker=dict(
            color='#E74C3C',
            line=dict(color='#000000', width=2.5)
        ),
        text=[f'{val:.2f}' for val in baseline],
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>Baseline</b><br>%{x}<br>Score: %{y:.2f}<extra></extra>',
        width=0.35
    ),
    go.Bar(
        name='AIMS (Proposed)',
        x=metrics,
        y=aims,
        marker=dict(
            color='#27AE60',
            line=dict(color='#000000', width=2.5)
        ),
        text=[f'{val:.2f}' for val in aims],
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>AIMS (Proposed)</b><br>%{x}<br>Score: %{y:.2f}<extra></extra>',
        width=0.35
    )
])

# ============================================================================
# UPDATE LAYOUT FOR CLARITY
# ============================================================================

fig.update_layout(
    title=dict(
        text=('Aggregate Quantitative Comparison Between Baseline and AIMS<br>'
              '<sub>Across Lexical, Semantic, and Factual Evaluation Metrics</sub>'),
        font=dict(size=18, color='black', family='Arial Black'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>Evaluation Metrics</b>',
        titlefont=dict(size=16, color='black', family='Arial Black'),
        tickfont=dict(size=13, color='black', family='Arial Black'),
        showgrid=False,
        showline=True,
        linewidth=2.5,
        linecolor='black',
        mirror=True
    ),
    yaxis=dict(
        title='<b>Score</b>',
        titlefont=dict(size=16, color='black', family='Arial Black'),
        tickfont=dict(size=13, color='black', family='Arial Black'),
        showgrid=True,
        gridwidth=1.5,
        gridcolor='lightgray',
        showline=True,
        linewidth=2.5,
        linecolor='black',
        mirror=True,
        range=[0, 1.15]
    ),
    barmode='group',
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=100, r=80, t=120, b=100),
    legend=dict(
        font=dict(size=14, color='black', family='Arial Black'),
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=2
    ),
    hovermode='x unified',
    width=1600,
    height=1000,
    font=dict(family='Arial')
)

# Add grid for Y axis
fig.update_yaxes(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.3)')

# ============================================================================
# SAVE INTERACTIVE HTML
# ============================================================================

fig.write_html('figure15_interactive.html')
print('✓ Interactive HTML figure saved as: figure15_interactive.html')
print('  → Open in browser for best clarity and interactivity')

# ============================================================================
# SAVE STATIC IMAGE
# ============================================================================

# Try to save as PNG using plotly (requires kaleido)
try:
    fig.write_image('figure15_plotly.png', width=1600, height=1000)
    print('✓ Static PNG saved as: figure15_plotly.png')
except:
    print('⚠ Plotly PNG export requires: pip install kaleido')
    print('  You can still use the HTML version for best results')

# Save as PDF
try:
    fig.write_image('figure15_plotly.pdf', width=1600, height=1000)
    print('✓ Static PDF saved as: figure15_plotly.pdf')
except:
    print('⚠ PDF export requires: pip install kaleido')

print('\n' + '='*70)
print('✅ INTERACTIVE FIGURE READY!')
print('='*70)
print('\n🎯 PLOTLY VERSION (MOST CLEAR):')
print('   • Open: figure15_interactive.html in your browser')
print('   • Hover over bars to see exact values')
print('   • Zoom, pan, and interact')
print('   • Crystal clear quality')
print('   • Perfect for presentations')
print('\n📊 STATIC VERSIONS:')
print('   • figure15_matplotlib.png/pdf (traditional academic)')
print('   • figure15_plotly.png/pdf (if kaleido installed)')
print('\n' + '='*70)
