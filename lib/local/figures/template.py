import json
from typing import Any
from plotly import subplots as sp, graph_objs as go

def BaseFigure(shape: tuple[int, int]=(1, 1), **kwargs) -> go.Figure:
    ncols, nrows = shape
    # column_widths=[0.1, 0.9], row_heights=[0.3, 0.1, 0.6],
    params: dict = dict(
        rows=nrows, cols=ncols,
        horizontal_spacing=0.02, vertical_spacing=0.02,
        shared_yaxes=True, shared_xaxes=True,
    ) | kwargs

    # override add_trace() to record subplot locations
    # so that we can see which subplot has a pie chart
    # since pie charts don't have axes and shift the whole order
    fig: Any = sp.make_subplots(**params)
    fig._subplot_locations = []
    original_add_trace = fig.add_trace
    def add_trace(*args, **kwargs):
        if "row" in kwargs and "col" in kwargs:
            x, y = kwargs["col"], kwargs["row"]
            fig._subplot_locations.append((type(args[0]), x, y))
        original_add_trace(*args, **kwargs)
    fig.add_trace = add_trace
    return fig

def ApplyTemplate(fig: go.Figure, default_xaxis: dict = dict(), default_yaxis: dict = dict(), axis: dict[str, dict] = dict(), layout: dict = dict()):
    # @axis
    # example: {"1 1 y": dict(showticklabels=True, categoryorder='array', categoryarray=cat_list)}
    # params: https://plotly.com/python/reference/layout/xaxis/

    color_none = 'rgba(0,0,0,0)'
    color_axis = 'rgba(0, 0, 0, 0.15)'
    DEF_AXIS: dict = dict(showgrid=False, showticklabels=True, linecolor="#212121", ticks="outside", gridcolor=color_axis, zerolinecolor=color_none, zerolinewidth=1)
    DEF_XAXIS: dict = DEF_AXIS|default_xaxis
    DEF_YAXIS: dict = DEF_AXIS|default_yaxis
    logged_cols, logged_rows = [], []
    _layout = layout.copy()
    _rows, _ncols = fig._get_subplot_rows_columns()
    nrows, ncols = [len(x) for x in [_rows, _ncols]]

    # pie charts don't have axes...
    # note down which trace types don't have axes
    problematic_traces = set()
    for d in fig["data"]:
        if "xaxis" in d or "yaxis" in d: continue
        problematic_traces.add(type(d))
    xfig: Any = fig
    problematic_positions = set()
    _spl = xfig._subplot_locations if hasattr(xfig, "_subplot_locations") else []
    for _t, x, y in _spl:
        if _t not in problematic_traces: continue
        problematic_positions.add((x, y))

    positions_to_skip = 0 # skip them when assigning axis
    for i in range(nrows*ncols):
        x, y = i%ncols + 1, i//ncols + 1
        i += 1
        ax = DEF_XAXIS | axis.get(f"{x} {y} x", DEF_XAXIS.copy())
        ay = DEF_YAXIS | axis.get(f"{x} {y} y", DEF_YAXIS.copy())
        if x in logged_cols: ax |= dict(type="log")
        if y in logged_rows: ay |= dict(type="log")
        if (x, y) in problematic_positions:
            positions_to_skip += 1
            continue
        axis_index = i-positions_to_skip if i != 1 else ''
        _layout[f"xaxis{axis_index}"] = ax
        _layout[f"yaxis{axis_index}"] = ay
    
    # print(json.dumps(_layout, indent=2))

    bg_col="white"
    W, H = 1000, 600
    _layout: dict = dict(
        width=W, height=H,
        paper_bgcolor=bg_col,
        plot_bgcolor=bg_col,
        margin=dict(
            l=15, r=15, b=15, t=15, pad=5
        ),
        font=dict(
            family="Arial",
            color="#212121",
            size=16,
        ),
        legend=dict(
            font=dict(
                size=12,
            ),
        ),
    ) | _layout
    fig.update_layout(**_layout)
    return fig
