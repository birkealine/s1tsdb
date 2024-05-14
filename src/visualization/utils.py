def add_text_box(
    ax, text, x=0.97, y=0.95, va="top", ha="right", bbox_props=dict(facecolor="wheat", boxstyle="round", alpha=0.9)
):
    d_kwargs = dict(transform=ax.transAxes, va=va, ha=ha, bbox=bbox_props)
    ax.text(x, y, text, **d_kwargs)
