
import itertools
import functools

import typing
from xml.etree.ElementInclude import XINCLUDE
import numpy
import pandas

import plotly.express
from plotly.express.colors import sample_colorscale

from . import rendering
from . import densities

import xtuples as xt


# ---------------------------------------------------------------

RENDERING = {None: None}
HTML = "HTML"

def set_rendering(val):
    RENDERING[None] = val

def return_chart(fig):
    if RENDERING[None] == HTML:
        return rendering.render_as_html(fig)
    return fig

# ---------------------------------------------------------------

def df_color_scale(df, col, color_scale):
    colors = sample_colorscale(
        color_scale, 
        numpy.linspace(0, 1, len(df[col].unique()))
    )
    color_map = {
        k: v for k, v in zip(
            sorted(df[col].unique()),
            colors,
        )
    }
    return color_map

# ---------------------------------------------------------------

def df_chart(
    df,
    x="date",
    y="value",
    title=None,
    color: typing.Optional[str]=None,
    discrete_color_scale=None,
    width=750,
    height=400,
    f_plot = plotly.express.line,
    fig=None,
    f_df = None,
    **kws,
):
    if f_df is not None:
        df = f_df(df)

    if color is not None:
        kws["color"] = color

    if discrete_color_scale is not None:
        kws["color_discrete_map"] = df_color_scale(
            df,
            color,
            discrete_color_scale
        )

    chart = f_plot(
        data_frame=df,
        x=x,
        y=y,
        title=title,
        **kws,
    )
    if fig is None:
        fig = chart
    else:
        fig.add_trace(chart.data[0])
        
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )

    return return_chart(fig)

df_line_chart = functools.partial(
    df_chart,
    f_plot = plotly.express.line,
    render_mode="svg",
    #
)

df_bar_chart = functools.partial(
    df_chart,
    f_plot = plotly.express.bar,
    #
)

df_scatter_chart = functools.partial(
    df_chart,
    f_plot = plotly.express.scatter,
    render_mode="svg",
    #
)

df_scatter_3d_chart = functools.partial(
    df_chart,
    f_plot = plotly.express.scatter_3d,
    # render_mode="svg",
    #
)

df_line_3d_chart = functools.partial(
    df_chart,
    f_plot = plotly.express.line_3d,
    # render_mode="svg",
    #
)

def f_df_density_df(gk, y, clip_quantile=.01):
    def f(df):
        gvs = df[gk].unique()
        vs = {
            gv: df[df[gk] == gv][y].values
            for gv in gvs
        }
        df = densities.gaussian_kde_1d_df(
            vs,
            key=gk,
            clip_quantile=clip_quantile,
        )
        return df
    return f

def df_density_chart(df, g, y, clip_quantile=.01, **kwargs):
    return df_line_chart(
        df,
        x="position",
        y="density",
        color=g,
        f_df = f_df_density_df(
            g, y, clip_quantile=clip_quantile
        ),
        **kwargs
    )

# ---------------------------------------------------------------

def df_facet_chart(
    df,
    x="date",
    y="value",
    title=None,
    facet=None,
    facet_row=None,
    facet_col=None,
    color=None,
    discrete_color_scale=None,
    share_y=False,
    share_x=False,
    width=750,
    height=400,
    fig=None,
    f_plot = plotly.express.line,
    f_df = None,
    **kws,
):
    if f_df is not None:
        df = f_df(df)

    if facet_row is None and facet is not None:
        assert facet_col is None, facet_col
        facet_row = facet

    if color is not None:
        kws["color"] = color

    if discrete_color_scale is not None:
        kws["color_discrete_map"] = df_color_scale(
            df,
            color,
            discrete_color_scale
        )

    chart = f_plot(
        data_frame=df,
        x=x,
        y=y,
        facet_row=facet_row,
        facet_col=facet_col,
        title=title,
        **kws,
    )
    if fig is None:
        fig = chart
    else:
        fig.add_trace(chart.data[0])

    if not share_x:
        fig.update_xaxes(matches=None, showticklabels=True)
    if not share_y:
        fig.update_yaxes(matches=None, showticklabels=True)

    fig.update_layout(
        autosize=False,
        width=width,
        height=height * len(df[facet_row].unique()),
    )

    return return_chart(fig)

df_facet_line_chart = functools.partial(
    df_facet_chart,
    f_plot = plotly.express.line,
    render_mode="svg",
    #
)

df_facet_bar_chart = functools.partial(
    df_facet_chart,
    f_plot = plotly.express.bar,
    #
)

df_facet_scatter_chart = functools.partial(
    df_facet_chart,
    f_plot = plotly.express.scatter,
    render_mode="svg",
    #
)

df_facet_scatter_3d_chart = functools.partial(
    df_facet_chart,
    f_plot = plotly.express.scatter_3d,
    render_mode="svg",
    #
)

df_facet_line_3d_chart = functools.partial(
    df_facet_chart,
    f_plot = plotly.express.line_3d,
    render_mode="svg",
    #
)

def df_density_facet_chart(df, g, y, clip_quantile=.01, **kwargs):
    return df_facet_line_chart(
        df,
        x="position",
        y="density",
        facet=g,
        f_df = f_df_density_df(
            g, y, clip_quantile=clip_quantile
        ),
        **kwargs
    )

def f_df_density_pair_df(columns, gk, clip_quantile=.01):
    def f(df):
        pairs = [
            pair for pair in itertools.combinations(columns, 2)
            if pair[0] != pair[1]
        ]
        vs = {
            ",".join(
                _c if isinstance(_c, str) else str(_c)
                for _c in [x, y]
            ): (
                df[x].values,
                df[y].values,
            )
            for x, y in pairs
        }
        df = densities.gaussian_kde_2d_df(
            vs,
            key=gk,
            clip_quantile=clip_quantile,
        )
        return df
    return f

def df_density_pair_chart(
    df,
    key="key",
    clip_quantile=.01,
    columns = None,
    excluding=xt.iTuple(),
    facet_col=None,
    separate=False,
    **kwargs
):
    if facet_col is not None:
        excluding = excluding.append(facet_col)

    if columns is None:
        columns = [
            col for col in df.columns if col not in excluding
        ]

    f_df = f_df_density_pair_df(
        columns, key, clip_quantile=clip_quantile
    )

    if facet_col is not None:
        by_v = {
            v: f_df(df[df[facet_col] == v])
            for v in df[facet_col].unique()
        }
        df = pandas.concat([
            sub_df.assign(**{
                facet_col: [v for _ in sub_df.index]
            }) for v, sub_df in by_v.items()
        ])
    else:
        df = f_df(df)
    chart_kws = dict(
        x="x",
        y="y",
        color="density",
        color_continuous_scale="Blues",
        height=400,
        width=600,
        **kwargs,
    )
    if not separate:
        return df_facet_scatter_chart(
            df,
            share_x=False,
            share_y=False,
            facet_row=key,
            facet_col=facet_col,
            **chart_kws,
        )
    return [
        df_scatter_chart(
            df[df[key] == v],
            **chart_kws,
        )
        for v in df[key].unique()
    ]

def vector_rays(nd, ns, gs, xs, ys, zs = None, i = 0, g = 0):
    assert nd.shape[-1] == 2 if zs is None else 3
    if len(nd.shape) == 1:
        nd = [nd]
    for ray in nd:
        xs.extend([0, ray[0]])
        ys.extend([0, ray[1]])
        if zs is not None:
            zs.extend([0, ray[2]])
        ns.extend([i, i])
        gs.extend([g, g])
        i += 1
    return i, len(nd)

import jax


def vector_ray_plot(
    vs,
    color = "n",
    _3d = False,
    **kws,
):
    
    ns = []
    gs = []
    xs = []
    ys = []
    rs = []
    cs = []
    zs = (None if not _3d else [])

    i = 0
    r = 0
    c = 0
    g = 0

    if isinstance(vs, (numpy.ndarray, jax.numpy.ndarray)):
        if len(vs.shape) == 2:
            vs = [vs]
        for g, _vs in enumerate(vs):
            i, m = vector_rays(_vs, ns, gs,xs, ys, zs = zs, i = i, g=g)
            for _ in range(m):
                rs.extend([r, r])
                cs.extend([c, c])

    elif isinstance(vs, (list, xt.iTuple, tuple)):
        if isinstance(vs[0], (numpy.ndarray, jax.numpy.ndarray)):
            vs = [vs]
        for r, rvs in enumerate(vs):
            for c, cvs in enumerate(rvs):
                if len(cvs.shape) == 2:
                    cvs = [cvs]
                for g, _cvs in enumerate(cvs):
                    i, m = vector_rays(_cvs, ns,gs, xs, ys, zs=zs, i = i, g=g)
                    for _ in range(m):
                        rs.extend([r, r])
                        cs.extend([c, c])
                    i = 0

    df = pandas.DataFrame(
        {
            **{
                "n": ns,
                "x": xs,
                "y": ys,
                "r": rs,
                "c": cs,
                "g": gs,
            },
            **({} if not _3d else {"z": zs})
        }
    )

    f_chart = (
        df_line_chart
        if len(set(rs)) == 1 and not _3d
        else df_facet_line_chart
        if not _3d
        else df_line_3d_chart
        if len(set(rs)) == 1
        else df_facet_line_3d_chart
    )

    return f_chart(
        df,
        x="x",
        y="y",
        color=color,
        **({} if _3d else dict(
            facet_row = (
                "r" if len(set(rs)) > 1 else None
            ),
            facet_col = (
                "c" if len(set(cs)) > 1 else None
            ),
        )),
        **({} if not _3d else dict(z="z")),
        **kws,
    )

# ---------------------------------------------------------------
