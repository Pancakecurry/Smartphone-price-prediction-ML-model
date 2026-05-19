import plotly.express as px

def apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            font_size=14,
            font_family="Inter",
            bordercolor="rgba(0,0,0,0.1)",
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig
