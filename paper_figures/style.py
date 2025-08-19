import matplotlib.colors
import matplotlib.ticker
import numpy as np
import seaborn as sns
from matplotlib.typing import ColorType
from suprtools.plotting import colors
from suprtools.plotting.palettes import cubehelix_doublegamma_palette
from suprtools.plotting.style import mode_palette
from suprtools.typing import AxvspanKwargs

from .constants import flaminia_q_range

# diagram_mirror_color: str = 'darkgoldenrod'
diagram_mirror_color: str = '#ccaf66'

mode_cmap = sns.cubehelix_palette(
    start=0.55, rot=0.2, hue=0.9, light=1, dark=0.3, as_cmap=True,
)

# old diverging colormap (purple: N=4; green: TEM00)
div_cmap = sns.diverging_palette(274, 160, s=70, l=75, center="dark", as_cmap=True)
'''Diverging colormap for mode mixing plots.'''

# new diverging colormap (purple: N=4; red: TEM00)
# for reference, Cardinal Red Light is H=352, S=52, L=47
# bumped the hue up to 12 to make the red end look more red
div_cmap_alt = sns.diverging_palette(274, 12, s=70, l=53, center="dark", as_cmap=True)
'''Alternative diverging colormap for mode mixing plots, with TEM00 mode in red.'''

colorbar_lower_lim = 5e-3
mode_mixing_cmap_norm = matplotlib.colors.FuncNorm(
    (lambda p: np.log(p / (1 - p)), lambda x: 1 / (1 + np.exp(-x))),
    vmin=colorbar_lower_lim,
    vmax=(1 - colorbar_lower_lim),
)
colorbar_kw = dict(
    ticks=matplotlib.ticker.LogitLocator(),
    # format=matplotlib.ticker.LogitFormatter(),
    format=matplotlib.ticker.StrMethodFormatter('{x:g}'),
)

alt_palette = sns.cubehelix_palette(
    n_colors=15, start=1, rot=6, reverse=True, gamma=0.8, dark=0.2, light=0.7,
)

def mode_colorfunc(n) -> ColorType:
    '''
    Parameters
    ----------
    n: int
        Transverse mode order.
    
    Returns
    -------
    ColorType
        Color specification understood by matplotlib
    '''
    return sns.color_palette('deep')[n//2 + 2]
    # does a swaparoo of some colors so TEM00 can be red.
    # inds = {
    #     0: 3,
    #     2: 2,
    #     4: 4,
    # }
    # ind = inds[n]
    # return sns.color_palette('deep')[ind]

plot_range_ghz: tuple[float, float] = (67, 105)

axvspan_kw: AxvspanKwargs = dict(
    facecolor='0.8',
    alpha=0.6,
)
# field_plot_gamma = 4
# field_plot_cmap = sns.cubehelix_palette(
#     start=-0.1,
#     rot=1.5,
#     gamma=field_plot_gamma,
#     hue=0.7,
#     light=(1 ** (1/field_plot_gamma)),
#     dark=(0.2 ** (1/field_plot_gamma)),
#     reverse=False,
#     as_cmap=True,
# )
field_plot_cmap = cubehelix_doublegamma_palette(
    start=2.70,
    rot=1.5,
    gamma=0.9,
    gamma_rot=9.1,
    hue=0.8,
    light=0.9,
    dark=0.15,
    as_cmap=True,
)

field_plot_log_mappable = matplotlib.cm.ScalarMappable(
    # norm=matplotlib.colors.LogNorm(vmax=3e+6, vmin=1e-4),  # when normalizing with \int |E|^2 dV = 1
    norm=matplotlib.colors.LogNorm(vmax=1e-10, vmin=3e-12),  # when normalizing to peak value
    # cmap=sns.cubehelix_palette(
    #     start=0,
    #     rot=0.4,
    #     gamma=1,
    #     hue=0.7,
    #     light=1.0,
    #     dark=(0.2),
    #     reverse=False, as_cmap=True,
    # ),
    cmap=field_plot_cmap,
)

field_plot_lin_mappable = matplotlib.cm.ScalarMappable(
    norm=matplotlib.colors.Normalize(
        vmax=0,
        vmin=-11,
    ),
    cmap=field_plot_cmap,
    # cmap='RdPu',
    # cmap=sns.cubehelix_palette(
    #     start=0, rot=0.4,
    #     hue=0.5, light=1.0, dark=0.2,
    #     reverse=False, as_cmap=True,
    # ),
)


def near_confocal_color(q):
    return alt_palette[q+1-20]


def less_confocal_color(q):
    return mode_palette(flaminia_q_range[1] - flaminia_q_range[0] + 1)[q - flaminia_q_range[0]]


limit_line_color = '0.5'


def make_paper_grids(ax):
    ax.grid(which='major', color='0.9')
    ax.grid(which='minor', color='0.95')


gemina_color = colors.stanford.cardinal_red_light
hadriana_color = colors.stanford.sky_light
