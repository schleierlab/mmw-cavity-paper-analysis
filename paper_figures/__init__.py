from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Optional, assert_never

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.legend_handler import HandlerTuple
from numpy.typing import NDArray
from scipy.constants import c, pi
from suprtools.cavity_loss import (
    Niobium,
    TemperatureFit,
    geom_factor_f,
    roughness_limit_finesse,
)
from suprtools.fp_theory.geometry import (
    FiniteMirrorSymmetricCavity,
    SymmetricCavityGeometry,
)
from suprtools.fp_theory.geometry.anticrossing import AnticrossingFit
from suprtools.gmsh_utils import CurlGradField
from suprtools.plotting import (
    AngleAnnotation,
    annotate_length,
    annotate_line,
    annotate_radius,
    label_subplots,
    latex_frexp10,
    mpl_usetex,
)
from suprtools.plotting.units import Units
from suprtools.rf import WideScanNetwork
from suprtools.rf.ringdown import RingdownCollectiveFit, RingdownSetSweep
from suprtools.typing import (
    ErrorbarKwargs,
    FillBetweenKwargs,
    PlotKwargs,
    TripcolorKwargs,
)
from uncertainties import UFloat
from uncertainties import unumpy as unp

from . import constants, style
from .fig2 import mixed_mode_finesse_plot


# TODO refactor this with fig 3 logic
def cavity_diagram(
        artist: FigureBase | Axes,
        target_geo: SymmetricCavityGeometry,
        target_mode: int,
        mirror_radius: float = 23e-3,
        annotate: bool = True,
        annotate_cavity: bool = True,
        draw_mirrors: bool = True,
        draw_cones: bool = True,
        x_range=None,
        z_range=None,
        pcolormesh_kw=dict(),
):
    ax: Axes
    if isinstance(artist, Axes):
        ax = artist
    elif isinstance(artist, FigureBase):
        ax = artist.subplots()
    else:
        assert_never(artist)

    xmax = mirror_radius
    zmax = target_geo.length / 2

    if x_range is None:
        x_range = np.linspace(-xmax, xmax, 201)
    if z_range is None:
        z_range = np.linspace(-zmax, zmax, 501)
    xs, zs = np.meshgrid(x_range, z_range)

    parax_freq = target_geo.paraxial_frequency(target_mode, 0)
    field = target_geo.paraxial_scalar_mode_field(xs, zs, parax_freq)

    pcolormesh_kwargs = dict(cmap=style.mode_cmap, rasterized=True) | pcolormesh_kw
    im = ax.pcolormesh(xs, zs, np.abs(field)**2, **pcolormesh_kwargs)
    im.set_rasterized(True)
    ax.set_aspect('equal')

    def upper_mirror_profile(r):
        return zmax - target_geo.mirror_curv_rad + np.sqrt(target_geo.mirror_curv_rad**2 - r**2)

    ax.set_ylim(-zmax, zmax)


    # z_R = pi w_0^2 / lambda = pi w_0^2 f / c
    # w_0 = sqrt(z0 c / (pi f))
    waist_w = np.sqrt(target_geo.z0 * c / (pi * parax_freq))
    
    if annotate:
        annotate_length(
            ax, '$w_0$', (0, 0), (-waist_w, 0),
            reverse=True,
        )
        annotate_length(ax, R'$w_0 \sqrt{2}$', (0, target_geo.z0), (-np.sqrt(2) * waist_w, target_geo.z0))
        annotate_length(ax, '$z_R$', (-1.5*waist_w, 0), (-1.5*waist_w, target_geo.z0))#, horizontalalignment='right', verticalalignment='center')


    if draw_mirrors:
        # TODO refactor with the coupling subfig plotting util
        mirror_center_thick = 5.08e-3
        mirror_back_z = zmax + mirror_center_thick

        for sign in [+1, -1]:
            ax.fill_between(
                x_range,
                sign * upper_mirror_profile(x_range),
                sign * mirror_back_z,
                color=style.diagram_mirror_color,
                # facecolor=matplotlib.colors.to_rgba(style.diagram_mirror_color, alpha=0.5),
                # edgecolor=matplotlib.colors.to_rgba(style.diagram_mirror_color, alpha=0.9),
                # linewidth=1.2,
            )
        ax.set_ylim(-mirror_back_z, mirror_back_z)

        if annotate_cavity:
            annotate_length(
                ax, '$L$',
                (-2*waist_w, -target_geo.length/2),
                (-2*waist_w, +target_geo.length/2),
                reverse=True,
            )
            # ax2.annotate("",
            #          xy=(0, 0), xycoords=ax1.transData,
            #          xytext=(0, 0), textcoords=ax2.transData,
            #          arrowprops=dict(arrowstyle="<->"))
            annotate_radius(
                ax,
                '$R$',
                (0, -zmax + target_geo.mirror_curv_rad),
                target_geo.mirror_curv_rad,
                np.deg2rad(-65),
                -10e-3,
            )

    if draw_cones:
        # imaging_na = 1/np.sqrt(2)
        # imaging_cone_slope = np.tan(np.arcsin(imaging_na))
        imaging_cone_slope = upper_mirror_profile(xmax) / xmax
        imaging_cone_xrange = x_range * 1
        ax.fill_between(
            imaging_cone_xrange,
            -imaging_cone_slope * imaging_cone_xrange,
            imaging_cone_slope * imaging_cone_xrange,
            color='0.8', alpha=0.3, linewidth=0,
        )

        # arrow_kw = dict(
        #     arrowstyle='Simple, tail_width=0.5, head_width=4, head_length=8',
        #     color='black',
        # )
        # radius = 10e-3
        # a3 = patches.FancyArrowPatch((10e-3, 0), (0, 10e-3), connectionstyle="arc3,rad=1.57", **arrow_kw)
        # ax.add_patch(a3)
        
        if annotate:
            AngleAnnotation(
                (0, 0), (1, 0), (1, imaging_cone_slope),
                ax=ax,
                size=40,
                unit='points',
                text=R'$\sin^{-1}\mathrm{NA}$',
                color='0.3',
                text_kw=dict(
                    ha='left',
                    va='center',
                    xytext=(5, 0),
                    fontsize='small',
                ),
                textposition='edge',
            )
    
    if annotate:
        max_ws = 1.4 if draw_cones else 0
        ax.plot(
            [-1.7 * waist_w, max_ws * waist_w], [0, 0],
            linestyle='solid',
            linewidth=1,
            color='0.3',
            alpha=0.5,
        )

    ax.axis('off')


def needed_mirror_rad_nosag(g, cavity_length_nom, wavelen, target_clipping_finesse):
    spot_size_w = np.sqrt((cavity_length_nom * wavelen / pi) / np.sqrt(1 - g**2))
    return spot_size_w * np.sqrt(1/2 * np.log(target_clipping_finesse / pi))


def spherical_sag(r, curv_rad):
    return curv_rad - np.sqrt(curv_rad**2 - r**2)


def parabolic_sag(r, curv_rad):
    return r**2 / (2 * curv_rad)


def root_finding_func_maker(g, *, target_clipping_finesse, wavelen, cavity_length, sag_func=spherical_sag):
    r_over_w_factor = np.sqrt(1/2 * np.log(target_clipping_finesse / pi))
    z0 = 1/2 * cavity_length * np.sqrt((1 + g) / (1 - g))
    waist_w0 = np.sqrt(wavelen * z0 / pi)
    mirror_curv_rad = cavity_length / (1 - g)
    
    def retfun(r):
        saggital_z = 1/2 * cavity_length - sag_func(r, mirror_curv_rad)
        beam_expansion_factor = np.sqrt(1 + (saggital_z / z0)**2)
        
        return r_over_w_factor * waist_w0 * beam_expansion_factor - r
    return retfun


def minimal_mirror_radius_scalarfunc(g, cavity_length, *, target_clipping_finesse, wavelen, sag_func, bracket_func=None):
    '''
    Assumes all inputs are scalars
    '''
    if bracket_func is None:
        def bracket_func(g, curv_rad, nosag_result):
            return curv_rad
        
    needed_radius_nosag = needed_mirror_rad_nosag(g, cavity_length, wavelen, target_clipping_finesse)
    mirror_curv_rad = cavity_length / (1 - g)
    
    opt_result = scipy.optimize.root_scalar(
        root_finding_func_maker(
            g,
            wavelen=wavelen,
            target_clipping_finesse=target_clipping_finesse,
            cavity_length=cavity_length,
            sag_func=sag_func,
        ),
        bracket=(0, bracket_func(g, mirror_curv_rad, needed_radius_nosag)),
    )
    return opt_result.root


minimal_mirror_radius = np.vectorize(minimal_mirror_radius_scalarfunc)


def maximal_na(g, cavity_length, *, target_clipping_finesse, wavelen, sag_func, bracket_func=None):
    curv_rads = cavity_length / (1 - g)
    min_mirror_radii = minimal_mirror_radius(
        g,
        cavity_length,
        target_clipping_finesse=target_clipping_finesse,
        wavelen=wavelen,
        sag_func=sag_func,
        bracket_func=bracket_func,
    )
    
    na_half_angle = np.arctan2(cavity_length / 2 - sag_func(min_mirror_radii, curv_rads), min_mirror_radii)
    return np.sin(na_half_angle)


def confocal_optimality_plot(
    fig: FigureBase,
    target_freq,
    target_geo,
    target_finesse=1e+10,
    xvals=np.linspace(-0.99, 0.99, num=199),
    xlabel=r'$g = 1 - \frac{L}{R}$',
    yvals=np.linspace(16, 80, num=129),
    ylabel='Length (mm)',
    gfunc=(lambda x, y: x),
    lfunc=(lambda x, y: y * 1e-3),  # plotted y value to length in [m]
    xfunc=(lambda g, length: g),
    yfunc=(lambda g, length: length * 1e+3),
    contours=np.arange(4, 40, 4),
    cut_x=0,
    colorbar_kw=dict(),
    gridspec_kw=dict(),
):
    """
    Parameters
    ----------
    gfunc, lfunc: (x: float, y: float) -> float
        Conversion functions from axis coordinates `(x, y)` to cavity parameters
        `(g, length)`, with `length` given in meters.
    xfunc, yfunc: (g: float, length: float) -> float
        Conversion functions from `(g, length)` (cavity parameters, with
        `length` given in meters) back to axis coordinates `(x, y)`.
    """
    axs = fig.subplots(
        nrows=2,
        ncols=2,
        sharex='col',
        sharey='row',
        gridspec_kw=(dict(height_ratios=[1,4], width_ratios=[4,1]) | gridspec_kw),
    )
    
    (ax_hcut, ax_ghost), (pcolor_ax, ax_vcut) = axs

    xmesh, ymesh = np.meshgrid(xvals, yvals)
    lengths_mesh = lfunc(xmesh, ymesh)
    g_mesh = gfunc(xmesh, ymesh)

    target_wavelen = c / target_freq
    maximal_imaging_nas = maximal_na(
        g_mesh,
        lengths_mesh,
        target_clipping_finesse=target_finesse,
        wavelen=target_wavelen,
        sag_func=spherical_sag,
    )

    na_plot = pcolor_ax.pcolormesh(
        xvals,
        yvals,
        maximal_imaging_nas,
        # cmap='cividis',
        cmap=sns.color_palette('flare_r', as_cmap=True),
        rasterized=True,
    )

    colorbar_kw_default = dict(
        location='top',
        # anchor=(0, 0.3),
        shrink=0.7,
    )
    colorbar_kwargs = colorbar_kw_default | colorbar_kw

    cbar = fig.colorbar(
        na_plot,
        # ax=axs,
        ax=(
            [ax_hcut, pcolor_ax]
            if colorbar_kwargs['location'] == 'top'
            else pcolor_ax
        ),
        **colorbar_kwargs,
    )
    if cbar.solids is None:
        # check to make mypy happy
        raise RuntimeError
    cbar.solids.set_rasterized(True)

    target_finesse_str = latex_frexp10(target_finesse)
    cbar.set_label(fr'Available numerical aperture for $F_\mathrm{{clip}} = {target_finesse_str}$')
    # ax.set_ylabel(r'$g = 1 - \frac{L}{R}$')
    # ax2.set_xlabel('Cavity length (mm)')

    target_x = xfunc(target_geo.g, target_geo.length)
    target_y = yfunc(target_geo.g, target_geo.length)
    pcolor_ax.scatter(
        target_x,
        target_y,
        s=100,
        marker='*',
        color='purple',
        zorder=5,
    )
    pcolor_ax.axhline(
        target_y,
        color='0.95',
        linestyle='dashed',
    )
    pcolor_ax.axvline(cut_x, color='0.95', linestyle='dashed')

    vcut_g = gfunc(cut_x, yvals)
    vcut_l = lfunc(cut_x, yvals)
    ax_vcut.plot(
        maximal_na(
            vcut_g,
            vcut_l,
            target_clipping_finesse=target_finesse,
            wavelen=target_wavelen,
            sag_func=spherical_sag,
        ),
        yvals,
    )
    
    cut_y = target_y
    hcut_g = gfunc(xvals, cut_y)
    hcut_l = lfunc(xvals, cut_y)
    ax_hcut.plot(
        xvals,
        maximal_na(
            hcut_g,
            hcut_l,
            target_clipping_finesse=target_finesse,
            wavelen=target_wavelen,
            sag_func=spherical_sag,
        ),
    )
    ax_vcut.set_xlabel('NA')
    ax_hcut.set_ylabel('NA')

    axs[-1, 0].set_ylabel(ylabel)
    axs[-1, 0].set_xlabel(xlabel)
    
    arrow_kw = dict(
        # transform=axs[-1, 0].transAxes,
        # ha="center", va="center",
        fontsize='small',
    )
    arrow_y = 0.15
    arrow_dx = 0.2
    def bbox_kw(direction: Literal['rarrow', 'larrow']):
        return dict(
            boxstyle=f'{direction},pad=0.05',
            fc="0.9",
            ec="0.9",
            lw=2,
        )

    pcolor_ax.text(
        cut_x + arrow_dx,
        arrow_y,
        "Concentric",
        transform=pcolor_ax.get_xaxis_transform(),
        va='center',
        ha='left',
        bbox=bbox_kw('rarrow'),
        **arrow_kw,
    )
    pcolor_ax.text(
        cut_x - arrow_dx,
        arrow_y,
        'Planar',
        transform=pcolor_ax.get_xaxis_transform(),
        va='center',
        ha='right',
        bbox=bbox_kw('larrow'),
        **arrow_kw,
    )

    arrow_x = 0.08
    # arrow_dy = 0.1

    pcolor_ax.text(
        arrow_x,
        0.9,
        "Paraxial",
        transform=pcolor_ax.transAxes,
        va='top',
        ha='center',
        bbox=bbox_kw('rarrow'),
        rotation=90,
        **arrow_kw,
    )
    pcolor_ax.text(
        arrow_x,
        0.1,
        'Divergent',
        transform=pcolor_ax.transAxes,
        va='bottom',
        ha='center',
        bbox=bbox_kw('larrow'),
        rotation=90,
        **arrow_kw,
    )

    # sslab_style(ax_hcut)
    # sslab_style(ax_vcut)
    ax_hcut.set_ylim(0, None)
    ax_vcut.set_xlim(0, None)
    # ax_ghost.set_adjustable('box')
    # ax_ghost.set_aspect(1, adjustable='box')
    ax_ghost.remove()
    # ax_hcut.set_box_aspect(1/3)
    # ax_vcut.set_box_aspect(3/1)
    # ax_ghost.set_box_aspect(1)
    # ax_ghost.axis('off')


    
    label_subplots(fig, [ax_hcut, ax_vcut], label_fmt='(roman)')

    length_contour = pcolor_ax.contour(
        xvals,
        yvals,
        lfunc(*np.meshgrid(xvals, yvals)) / target_wavelen,
        levels=contours,
        linewidths=1,
        colors='aliceblue',
        alpha=0.5,
        # locator=ticker.MultipleLocator(10),
    )
    pcolor_ax.clabel(
        length_contour,
        contours,
        inline=True,
        fmt=R'$L/\lambda = %.0f$',
        fontsize='small',
    )
    ax_vcut.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_vcut.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_vcut.xaxis.set_major_locator(ticker.AutoLocator())
    ax_hcut.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    for ax in (ax_hcut, ax_vcut):
        style.make_paper_grids(ax)
        # ax.grid(which='major', color='0.9')
        # ax.grid(which='minor', color='0.95')

    return pcolor_ax, ax_hcut, ax_vcut, cbar



def single_network_vf_cartesian(network: WideScanNetwork, ax: Axes):
    vf = network.fit_network()
    uparams = vf.closest_pole_uparams(
        network.frequency.center, frequency_err_max=400e3
    )
    resmag_cyc = unp.sqrt(uparams[2] ** 2 + uparams[3] ** 2) / (2 * pi)

    if mpl_usetex():
        label_omega = rf"$\omega_0 = 2\pi\times \SI{{{uparams[1]/(2*pi*1e+9):S}}}{{\GHz}}$"
    else:
        label_omega = rf"$\omega_0 = 2\pi\times {uparams[1]/(2*pi*1e+9):S}$ GHz"
    vf.plot_model_cartesian(
        ax=ax,
        data_plot_kw=dict(
            marker='.',
            markersize=0.5,
            linewidth=0.3,
        ),
        model_plot_kw=dict(
            linestyle="None",
            marker=".",
            color="red",
            markersize=1,
            label="\n".join(
                [
                    label_omega,
                    rf"$\kappa = 2\pi\times ${vf._format_ufreq(-uparams[0]/pi)}",
                    rf"$|A|\sqrt{{\kappa_1\kappa_2}} = 2\pi\times ${vf._format_ufreq(resmag_cyc)}",
                ]
            ),
        ),
        scale=1e3,
    )

    # ax.legend(fontsize="x-small")
    if mpl_usetex():
        ax.set_xlabel(R"$10^3 \times \operatorname{Re}(S_{21})$")
        ax.set_ylabel(R"$10^3 \times \operatorname{Im}(S_{21})$")
    else:
        ax.set_xlabel(R"$10^3 \times \mathrm{Re}(S_{21})$")
        ax.set_ylabel(R"$10^3 \times \mathrm{Im}(S_{21})$")
    ax.set_box_aspect(1)
    ax.set_aspect(1)


def q_circle_figure(plot_networks: Sequence[WideScanNetwork], fig=None, axs=None):
    if (fig is None) != (axs is None):
        raise ValueError
    if fig is None:
        fig, axs = plt.subplots(ncols=(1 + len(plot_networks)), layout="constrained")
    else:
        if len(axs) != 1 + len(plot_networks):
            raise ValueError

    circle_center = np.asarray((3, 4))
    residue_normed = complex(*circle_center)

    def pole_response(detuning_per_kappa, circle_radius_cmplx):
        return circle_radius_cmplx / (0.5 + 1j * np.asarray(detuning_per_kappa))

    for resolution, kwargs in [
        (201, dict(marker=".", markersize=5, linestyle="None")),
        (2001, dict(marker=None, linestyle="solid")),
    ]:
        detuning_per_kappa = np.linspace(
            -10, +10, num=resolution
        )  # (\omega - \omega_0) / \kappa
        pole_response_points = pole_response(detuning_per_kappa, residue_normed)
        axs[0].plot(
            np.real(pole_response_points),
            np.imag(pole_response_points),
            color="C0",
            **kwargs,
        )

    marked_points_cmplx = pole_response([-0.5, 0, +0.5], residue_normed)
    marked_points = np.array(
        [np.real(marked_points_cmplx), np.imag(marked_points_cmplx)]
    ).T
    axs[0].plot(
        marked_points[:, 0],
        marked_points[:, 1],
        marker=".",
        markersize=10,
        color="red",
        linestyle="None",
    )
    annotate_length(
        axs[0],
        R"$2a/\kappa$",
        (0, 0),
        (2 * np.real(residue_normed), 2 * np.imag(residue_normed)),
        arrowprops=dict(arrowstyle="<|-", shrinkA=4, shrinkB=0),
        offset_points=12,
    )
    point_annots = [R"-\frac{\kappa}{2}", "", R"+\frac{\kappa}{2}"]
    for i, point in enumerate(marked_points):
        axs[0].annotate(
            Rf"$\omega = \omega_0 {point_annots[i]}$",
            point,
            xytext=(point - circle_center) * (+1 if i == 1 else -1),
            ha=("right" if i == 2 else "left"),
            textcoords="offset points",
        )

    for i in range(len(plot_networks)):
        single_network_vf_cartesian(plot_networks[i], axs[i + 1])

    if mpl_usetex():
        axs[0].set_xlabel(R"$\operatorname{Re}(H_*(i\omega))$")
        axs[0].set_ylabel(R"$\operatorname{Im}(H_*(i\omega))$")
    else:
        axs[0].set_xlabel(R"$\mathrm{Re}(H_*(i\omega))$")
        axs[0].set_ylabel(R"$\mathrm{Im}(H_*(i\omega))$")
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[0].tick_params(labelbottom=False, labelleft=False)

    axs[0].set_aspect(1)
    fig.align_labels()


def finesse_subplot(
        ax: Axes,
        rdset_sweep: RingdownSetSweep,
        probe = None,
        probe_z: Optional[float] = None,
        extrapolation_r: Optional[float] = None,
        legend_kw: Optional[dict] = dict(),
        kwarg_func: Optional[Callable[[Any, int, int], PlotKwargs]] = None,
        errorbar_kw=dict(),
        plot_modes=[21, 25, 27, 29],
        annotate_lines=True,
):
    ebar_conts = rdset_sweep.plot(
        ax,
        plot_modes=plot_modes,
        probe=probe,
        probe_z=probe_z,
        extrapolation_r=extrapolation_r,
        kwarg_func=kwarg_func,
        annotate_lines=annotate_lines,
        **errorbar_kw,
    )

    ax.set_ylabel('Finesse')
    ax.set_yscale('log')

    handles = list(ebar_conts.values())
    labels = [fr'$\mathrm{{TEM}}_{{{q}, 0, 0, \{{x, y\}}}}$' for q in plot_modes]

    if legend_kw is not None:
        legend_kwargs = dict(fontsize='x-small', ncol=1) | legend_kw
        ax.legend(
            handles,
            labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            **legend_kwargs,
        )


def parse_quadrant_str(s: str):
    ystr, xstr = s.lower().split()
    match ystr:
        case ('lower' | 'bottom'):
            ysign = -1
        case ('upper' | 'top'):
            ysign = +1
        case _:
            raise ValueError
    match xstr:
        case 'left':
            xsign = -1
        case 'right':
            xsign = +1
        case _:
            raise ValueError

    return (xsign, ysign)


def field_plot(
        ax: Axes,
        nodes,
        normalized_field_cmplx,
        quadrant,
        scaling=1,
        scale='log',
        mirror_thick=1.415e-3,  # TODO remove/refactor some default params?
        mirror_radius=24e-3,
        probe_offset=constants.probe_offset,
        cavity: Optional[FiniteMirrorSymmetricCavity] = None,
        probe_r=None,
        probe_line_kw=dict(),
        **kwargs,
):
    signs = parse_quadrant_str(quadrant)

    nonneg_mask = (nodes[:, 1] >= 0)
    nonneg_nodes = nodes[nonneg_mask]
    
    default_kw: TripcolorKwargs
    normalized_intensity = (np.abs(normalized_field_cmplx[nonneg_mask])**2).sum(axis=1)
    if scale == 'log':
        default_kw = TripcolorKwargs(
            cmap=style.field_plot_lin_mappable.cmap,
            norm=style.field_plot_lin_mappable.norm,
            edgecolors='None',
            shading='gouraud',
        )
        plot_intensity = np.log10(normalized_intensity)
    elif scale == 'lin':
        default_kw = TripcolorKwargs(
            cmap=style.field_plot_lin_mappable.cmap,
            norm=style.field_plot_lin_mappable.norm,
            shading='gouraud',
        )
        plot_intensity = normalized_intensity
    else:
        raise ValueError
    tripcolor_kw: TripcolorKwargs = dict(rasterized=True) | default_kw | kwargs

    tripcolor = ax.tripcolor(
        signs[0] * nonneg_nodes[:, 0] * scaling,
        signs[1] * nonneg_nodes[:, 1] * scaling,
        plot_intensity,
        **tripcolor_kw,
    )

    if cavity is not None:
        half_length = cavity.geometry.length / 2
        r_range = np.linspace(0, mirror_radius)
        mirror_back_z = half_length + mirror_thick
        mirror_style = FillBetweenKwargs(
            # facecolor='white',
            # edgecolor=style.diagram_mirror_color,
            color=style.diagram_mirror_color,
        )

        ax.fill_between(
            signs[0] * scaling * r_range,
            signs[1] * scaling * unp.nominal_values(cavity.mirror_z(r_range)),
            signs[1] * scaling * unp.nominal_values(mirror_back_z),
            **mirror_style,
        )
        
        if probe_r is not None:
            mirror_edge_z = cavity.edge_z
            # mirror_sag = mirror_curvature - np.sqrt(mirror_curvature**2 - mirror_radius**2)

            # probe_line_default_kw = PlotKwargs(color='red', linestyle='dashed')
            probe_z_scaled = (mirror_edge_z - probe_offset) * scaling * signs[1]
            # ax.plot(
            #     np.array(probe_r) * scaling * signs[0],
            #     [probe_z] * 2,
            #     **(probe_line_default_kw | probe_line_kw),
            # )

            probe_r_scaled = np.asarray(probe_r) * scaling * signs[0]
            ax.annotate(
                '',
                xy=(probe_r_scaled[0], probe_z_scaled),
                xytext=(probe_r_scaled[1], probe_z_scaled),
                xycoords='data',
                arrowprops=dict(arrowstyle='|-|, widthA=0.25, widthB=0.25', **probe_line_kw),
            )
    ax.set_aspect(1)
    ax.autoscale(tight=True)

    return tripcolor


def field_plot_linlog(
        ax,
        nodes,
        normalized_field_cmplx,
        scaling=1,
        lin_quadrant='upper right',
        log_quadrant='lower right',
        cavity: Optional[FiniteMirrorSymmetricCavity] = None,
        mirror_radius=None,
        mirror_thick=None,
        probe_r=[16e-3, 24e-3],
):
    # nonneg_mask = (nodes[:, 1] >= 0)
    # nonneg_nodes = nodes[nonneg_mask]

    loghalf = field_plot(
        ax, nodes, normalized_field_cmplx,
        quadrant=log_quadrant, scale='log', scaling=scaling,
        cavity=cavity,
        mirror_thick=mirror_thick,
        mirror_radius=mirror_radius,
        probe_r=probe_r,
    )

    linhalf = None
    if lin_quadrant is not None:
        linhalf = field_plot(
            ax,
            nodes,
            normalized_field_cmplx,
            quadrant=lin_quadrant,
            scale='lin',
            scaling=scaling,
            cavity=cavity,
            mirror_thick=mirror_thick,
            mirror_radius=mirror_radius,
        )
    return loghalf, linhalf


def field_plot_inset(
    ax,
    simfield: CurlGradField,
    cavity: FiniteMirrorSymmetricCavity,
    quadrant,
    probe_r,
    probe_offset,
    probe_line_kw=dict(),
    scaling=1e+3,
    field_scaling=1
):
    nodes = simfield.nodes
    norm_field = simfield.e_field
    
    field_plot(
        ax,
        nodes,
        norm_field * field_scaling,
        quadrant,
        cavity=cavity,
        scaling=scaling,
        scale='log',
        probe_r=probe_r,
        probe_offset=probe_offset,
        probe_line_kw=probe_line_kw,
    )

    _, ysign = parse_quadrant_str(quadrant)
    if ysign == +1:
        ax.set_ylim(0, scaling * cavity.back_z)
    elif ysign == -1:
        ax.set_ylim(-scaling * cavity.back_z, 0)


# TODO refactor into FiniteMirrorSymmetricCavity?
def finesse_model(freq, limit_finesse, fudge_factor, cav: FiniteMirrorSymmetricCavity):
    geo = cav.geometry
    
    waist = np.sqrt(unp.nominal_values(geo.z0) * c / (pi * freq))
    spot_size_withsag = waist * np.sqrt(1 + (unp.nominal_values(cav.edge_z / geo.z0))**2)
    one_way_clipping_loss = np.exp(- 2 * cav.mirror_radius**2 / (fudge_factor * spot_size_withsag)**2)
    clipping_finesse = pi / one_way_clipping_loss

    return 1 / (1 / clipping_finesse + 1 / limit_finesse)

fig4a_model = lmfit.Model(finesse_model, independent_vars=['freq'], param_names=['limit_finesse', 'fudge_factor'])


def ringdown_plot(
        ax,
        collective_fit: RingdownCollectiveFit,
        data_kw: PlotKwargs = dict(),
        model_kw: PlotKwargs = dict(),
        max_t=30e-3,
): 
    collective_fit.plot_fit(
        ax,
        xscale=1e+3,
        xrange=(None, max_t),
        normalized=True,
        data_kw=data_kw,
        model_kw=model_kw,
        noise_removed=True,
    )

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power fraction')
    ax.set_yscale('linear')


def single_ringdown_inset(inset_ax, ringdown_collective_fit, data_kw: PlotKwargs = dict()):
    ringdown_plot(inset_ax, ringdown_collective_fit, data_kw=data_kw)
    inset_ax.tick_params(labelsize='small')
    inset_ax.xaxis.label.set_size('small')
    inset_ax.yaxis.label.set_size('small')
    inset_ax.get_legend().set_visible(False)
    inset_ax.patch.set_alpha(0.8)


def fit_fig4_model(cav, highest_fins_array):
    params = fig4a_model.make_params(
        limit_finesse=dict(value=6e+7, min=10, max=1e+9),
        fudge_factor=dict(value=1.3, min=1, max=3),
    )

    freq = highest_fins_array['freq']
    finesses_u = highest_fins_array['finesse']
    return fig4a_model.fit(
        unp.nominal_values(finesses_u),
        weights=1/(unp.std_devs(finesses_u)),
        freq=freq,
        params=params,
        cav=cav,
        nan_policy='omit',
    )


def highest_finesse_subplot(
        ax: Axes,
        rdset_sweeps: Sequence[tuple[RingdownSetSweep, FiniteMirrorSymmetricCavity, Callable[[int, int], bool], Mapping[str, Any]]],
        uparams,
        temperature: float,
        warm_lambda: bool,
        impure_lambda: bool,
        impure_xi: bool,
        roughness_rms: Optional[float] = None,
        minimal_limit_labels: bool = False,
        avoided_crossing_fit: Optional[AnticrossingFit] = None,
        vortex_rrr: Optional[float] = None,
        magnetic_flux_density: Optional[float | UFloat] = None,
        mean_defect_depth=100e-9,
        plot_range_ghz: tuple[float, float] = (67, 106),
        vortex_offset_pts: tuple[float, float] = (0, 3),
        roughness_offset_pts: tuple[float, float] = (0, 3),
        cmap=style.div_cmap,
        ylim: Optional[tuple[float, float]] = None,
):
    if avoided_crossing_fit is not None:
        mixed_mode_finesse_plot(ax, avoided_crossing_fit, uparams, cmap=cmap)

    freq_space_ghz = np.linspace(*plot_range_ghz, num=101)
    
    for rdset_sweep, cav, mask, kwargs in rdset_sweeps:
        rdset_sweep.plot_highest_finesse(ax, mask=mask, **kwargs)
        highest_fins = rdset_sweep.highest_finesse_values(mask=mask)

        fit = fit_fig4_model(cav, highest_fins)
        ax.plot(
            freq_space_ghz,
            fit.eval(freq=(freq_space_ghz * 1e+9)),
            **kwargs,
        )
        print(fit.params)

    annotations_to_do = []

    if roughness_rms is not None:
        ax.plot(
            freq_space_ghz,
            roughness_limit_finesse(1e+9 * freq_space_ghz, roughness_rms),
            color=style.limit_line_color,
            linestyle='dashed',
        )
        ax.set_yscale('log')
        roughness_text = (
            'roughness'
            if minimal_limit_labels else
            f'{roughness_rms * 1e+9:.0f} nm roughness limit'
        )
        def annotate_roughness():
            annotate_line(
                ax,
                text=roughness_text,
                annotation_x=freq_space_ghz.mean(),
                line_func=(lambda f_ghz: roughness_limit_finesse(1e+9 * f_ghz, roughness_rms)),
                color=style.limit_line_color,
                offset_pts=roughness_offset_pts,
                horizontalalignment='center',
                verticalalignment='baseline',
                fontsize='small',
            )
        
        annotations_to_do.append(annotate_roughness)

    if (vortex_rrr is not None) and (magnetic_flux_density is not None):
        niobium = Niobium(
            residual_resistivity_ratio=vortex_rrr,
            warm_lambda=warm_lambda,
            impure_lambda=impure_lambda,
            impure_xi=impure_xi,
        )

        # exponential distribution with the following EV:
        segment_length_ev = mean_defect_depth

        # riemann sum boxwidth
        d_seg_len = segment_length_ev / 100
        segment_length_range = np.linspace(d_seg_len, 10000 * d_seg_len, num=10000)
        probability_masses = d_seg_len \
            * np.exp(-segment_length_range / segment_length_ev) / segment_length_ev

        def limiting_finesse(freqs_ghz: NDArray) -> NDArray:
            # shape: (n_freqs, n_seglens)
            integration_arr: NDArray = niobium.trapped_vortex_resistance_per_field(
                freqs_ghz[..., np.newaxis] * 1e+9,
                segment_length_range,
                temperature=temperature,
                method='exact',
            )
            r_per_b_ev = np.sum(probability_masses * integration_arr, axis=-1)

            # shape: (n_freqs,)
            resistance_expected_value = magnetic_flux_density * r_per_b_ev
            return geom_factor_f / resistance_expected_value

        vortex_lim_line_kw = PlotKwargs(
            color=style.limit_line_color,
            linestyle='dashdot',
        )
        if roughness_rms is not None:
            def vortex_plus_roughness_finesse(freq_ghz):
                vortex_fin = unp.nominal_values(limiting_finesse(freq_ghz))
                roughness_fin = roughness_limit_finesse(1e+9 * freq_ghz, roughness_rms)
                return 1 / (1 / vortex_fin + 1 / roughness_fin)

            ax.plot(
                freq_space_ghz,
                vortex_plus_roughness_finesse(freq_space_ghz),
                **vortex_lim_line_kw,
            )
            annotation_field = f'{unp.nominal_values(magnetic_flux_density).item() * 1e+4 :.2f}'
            vortex_label = (
                'roughness + vortex'
                if minimal_limit_labels else
                f'roughness loss + vortex loss at {annotation_field} G field'
            )
            def annotate_vortices():
                annotate_line(
                    ax,
                    vortex_label,
                    # f'vortex loss at {magnetic_flux_density*1e+4:.1f} G field (RRR = {round(vortex_rrr, -1):.0f})',
                    annotation_x=freq_space_ghz.mean(),
                    line_func=vortex_plus_roughness_finesse,
                    # line_func=(lambda arr: np.full_like(arr, lim_finessse)),
                    color=style.limit_line_color,
                    offset_pts=vortex_offset_pts,
                    horizontalalignment='center',
                    verticalalignment='baseline',
                    fontsize='small',
                )
        else:
            if isinstance(magnetic_flux_density, float):
                ax.plot(
                    freq_space_ghz,
                    limiting_finesse(freq_space_ghz),
                    **vortex_lim_line_kw,
                )

                annotation_field = f'{magnetic_flux_density * 1e+4 :.1f}'
                annotation_linefunc = limiting_finesse
            elif isinstance(magnetic_flux_density, UFloat):
                limit_finesse_n = unp.nominal_values(limiting_finesse(freq_space_ghz))
                limit_finesse_s = unp.std_devs(limiting_finesse(freq_space_ghz))
                ax.fill_between(
                    freq_space_ghz,
                    limit_finesse_n - limit_finesse_s,
                    limit_finesse_n + limit_finesse_s,
                    color='0.5',
                    alpha=0.5,
                    linewidth=0,
                )

                annotation_field = f'{magnetic_flux_density * 1e+4 :.2S}'
                def annotation_linefunc(freqspace):
                    return unp.nominal_values(limiting_finesse(freqspace)) \
                        + unp.std_devs(limiting_finesse(freqspace))

            def annotate_vortices():
                annotate_line(
                    ax,
                    f'vortex loss at {annotation_field} G field',
                    # f'vortex loss at {magnetic_flux_density*1e+4:.1f} G field (RRR = {round(vortex_rrr, -1):.0f})',
                    min(freq_space_ghz),
                    line_func=annotation_linefunc,
                    # line_func=(lambda arr: np.full_like(arr, lim_finessse)),
                    color=style.limit_line_color,
                    offset_pts=vortex_offset_pts,
                    horizontalalignment='left',
                    verticalalignment='baseline',
                    fontsize='small',
                )
        annotations_to_do.append(annotate_vortices)

    ax.set_ylim(ylim)

    # these have to happen at the end, once axis limits are no longer changing
    for annotation_func in annotations_to_do:
        annotation_func()


def segment_length_dep_plot(
        rrr,
        freqs,
        segment_lengths,
        magnetic_flux_density,
        temperature: float,
        impure_lambda: bool,
        impure_xi: bool,
        vspan: Optional[tuple[float, float]] = None,
        ax: Optional[Axes] = None,
):
    plot_ax: Axes
    if ax is None:
        _, plot_ax = plt.subplots()
        plot_ax.set_xlabel(Rf'Frequency $\omega/2\pi$ ({Units.HZ.mplstr()})')
    else:
        plot_ax = ax

    def resistance_per_field_to_finesse(r_per_b):
        '''ohms/tesla --> finesse, or finesse --> ohms/tesla'''
        return geom_factor_f / (r_per_b * magnetic_flux_density)
    def secax_function(x):
        # extra 1e-4 so we convert between finesse and ohms/gauss
        return resistance_per_field_to_finesse(x * 1e+4)
    secax = plot_ax.secondary_yaxis('right', functions=(secax_function,)*2)
    if ax is None:
        if mpl_usetex():
            plot_ax.set_ylabel(Rf'Limiting finesse at $B = \SI{{{1e+4 * magnetic_flux_density}}}{{\gauss}}$')
            secax.set_ylabel(R'$R_\text{vortex}(\omega)/B$ (\si{\ohm/\gauss})')
        else:
            plot_ax.set_ylabel(f'Limiting finesse at $B = {1e+4 * magnetic_flux_density}$ G')
            secax.set_ylabel(R'$R_\text{vortex}(\omega)/B$ ($\Omega$/G)')

    nb_impure = Niobium(
        residual_resistivity_ratio=rrr,
        warm_lambda=True,
        impure_lambda=impure_lambda,
        impure_xi=impure_xi,
    )


    vortex_palette = sns.cubehelix_palette(start=.5, rot=-0.5, light=0.8, dark=0.3, n_colors=len(segment_lengths))
    for i, segment_length in enumerate(segment_lengths):
        if mpl_usetex():
            label = Rf'$d = \SI{{{segment_length * 1e+9:.3g}}}{{\nm}}$'
        else:
            label = Rf'$d = {segment_length * 1e+9:.3g}$ nm'
        plot_ax.plot(
            freqs,
            resistance_per_field_to_finesse(
                nb_impure.trapped_vortex_resistance_per_field(
                    freqs,
                    segment_length,
                    temperature,
                    method='exact',
                ),
            ),
            color=vortex_palette[i],
            label=label,
        )
        # ax.axvline(nb_impure.trapped_vortex_char_freq_ell(segment_length), color=vortex_palette[i])

    charlens = segment_lengths
    vortex_average_palette = sns.cubehelix_palette(start=2, rot=-0.5, light=0.8, dark=0.3, n_colors=len(charlens), reverse=True)
    for i, charlen in enumerate(charlens):
        npts = 10000
        segment_lengths_int = np.linspace(charlen/100, 100*charlen, num=npts)
        dl = charlen * 100 / npts
        prob_dist = np.exp(-segment_lengths_int / charlen) / charlen * dl


        if mpl_usetex():
            label = Rf'$\overline{{d}} = \SI{{{charlen * 1e+9:.3g}}}{{\nm}}$'
        else:
            label = Rf'$\overline{{d}} = {charlen * 1e+9:.3g}$ nm'

        plot_ax.plot(
            freqs,
            resistance_per_field_to_finesse(
                (
                    nb_impure.trapped_vortex_resistance_per_field(
                        freqs[..., np.newaxis],
                        segment_lengths_int,
                        temperature,
                    ) * prob_dist
                ).sum(axis=-1)
            ),
            color=vortex_average_palette[i],
            label=label,
            linestyle='dotted',
        )

    vortex_characteristic_freq_lambda = nb_impure.trapped_vortex_char_freq_lambda(
        temperature,
    )
    plot_ax.axvline(vortex_characteristic_freq_lambda, linestyle='dashdot', color='0.5')
    high_freq_lim_r_per_b = nb_impure.trapped_vortex_resistance_per_field_highfreqlim(
        temperature,
    )
    high_freq_lim_plot = resistance_per_field_to_finesse(high_freq_lim_r_per_b)
    plot_ax.axhline(
        high_freq_lim_plot,
        linestyle='dashed',
        color='0.5',
    )

    high_freq_lim_expr = R'\frac{\pi\xi^2\rho_n}{\lambda\Phi_0}'
    high_freq_lim_uohm_per_gauss = high_freq_lim_r_per_b * 1e+6 / 1e+4
    high_freq_lim_str = (
        Rf'${high_freq_lim_expr} = \SI{{{high_freq_lim_uohm_per_gauss:.0f}}}{{\micro\ohm/\gauss}}$'
        if mpl_usetex() else
        Rf'${high_freq_lim_expr} = {high_freq_lim_r_per_b * 1e+6 / 1e+4:.0f}$ $\mu\Omega/\mathrm{{G}}$'
    )
    plot_ax.annotate(
        high_freq_lim_str,
        xy=(max(freqs), high_freq_lim_plot),
        xycoords=plot_ax.transData,
        xytext=(-3, 3),
        textcoords='offset points',
        ha='right',
        va='bottom',
        fontsize='small',
    )

    omega_lambda_expr = R'\omega_\lambda = \frac{g\rho_n \xi^2 }{ 2\mu_0 \lambda^4 }'
    omega_lambda_ghz = vortex_characteristic_freq_lambda / 1e+9
    omega_lambda_str = (
        Rf'${omega_lambda_expr} = 2\pi\times \SI{{{omega_lambda_ghz:.0f}}}{{\GHz}}$'
        if mpl_usetex() else
        Rf'${omega_lambda_expr} = 2\pi\times {omega_lambda_ghz:.0f}$ GHz'
    )
    plot_ax.annotate(
        omega_lambda_str,
        xy=(vortex_characteristic_freq_lambda, 1),
        xycoords=plot_ax.get_xaxis_transform(),
        xytext=(-3, -5),
        textcoords='offset points',
        rotation=90,
        ha='right',
        va='top',
        fontsize='small',
    )

    if vspan is not None:
        plot_ax.axvspan(*vspan, **style.axvspan_kw)

    plot_ax.set_xscale('log')
    plot_ax.set_yscale('log')
    plot_ax.legend(ncols=1, fontsize='small')
    return plot_ax, secax


def vortex_loss_rrr_dep_plot(
        exemplar_rrr,
        freqs,
        mean_segment_length: float,
        temperature: float,
        impure_lambda: bool,
        impure_xi: bool,
        vspan: Optional[tuple[float, float]] = None,
        ax: Optional[Axes] = None,
):
    '''
    Parameters
    ----------
    temperature: scalar
        Temperature for calculating temperature-dependent penetration depth.
        To disable temperature dependence, simply pass in zero.
    '''
    plot_ax: Axes
    if ax is None:
        _, plot_ax = plt.subplots()
        plot_ax.set_xlabel(R'Frequency $\omega/2\pi$ (Hz)')
    else:
        plot_ax = ax
    generic_rrrs = np.geomspace(10, 1000, num=5)[:-1]

    palette = sns.cubehelix_palette(
        n_colors=len(generic_rrrs),
        start=1.8,
        rot=0.7,
        gamma=2,
        hue=1,
        light=0.9,
        dark=0.4,
    )

    rrrs = [exemplar_rrr] + list(generic_rrrs)
    for i, rrr in enumerate(rrrs):
        niobium = Niobium(
            residual_resistivity_ratio=rrr,
            warm_lambda=True,
            impure_lambda=impure_lambda,
            impure_xi=impure_xi,
        )

        charlen = mean_segment_length
        npts = 10000
        segment_lengths_int = np.linspace(charlen/100, 100*charlen, num=npts)
        dl = charlen * 100 / npts
        prob_dist = np.exp(-segment_lengths_int / charlen) / charlen * dl

        color = ('red' if i == 0 else palette[i-1])

        plot_ax.plot(
            freqs,
            (niobium.trapped_vortex_resistance_per_field(freqs[..., np.newaxis], segment_lengths_int) * prob_dist).sum(axis=-1) / 1e+4,
            # color=vortex_average_palette[i],
            label=fR'$\ell\sim\mathrm{{Exp}}({charlen * 1e+9:.0f})$ nm',
            color=color,
            # linestyle='dotted',
        )
        annotate_line(
            plot_ax,
            ('RRR = ' if i == 0 else '') + f'{rrr:.3g}',
            annotation_x=freqs[-1],
            line_func=(lambda x: np.full_like(x, niobium.trapped_vortex_resistance_per_field_highfreqlim(temperature) / 1e+4)),
            offset_pts=(0, 2),
            horizontalalignment='right',
            color=color,
            fontsize='small',
        )

        vortex_characteristic_freq_lambda = niobium.trapped_vortex_char_freq_lambda(
            temperature,
        )
        plot_ax.axvline(vortex_characteristic_freq_lambda, linestyle='dashdot', color=color)
        plot_ax.annotate(
            '$'
                + (R'\omega_\lambda = \frac{{g\rho_n \xi^2 }}{{ 2\mu_0 \lambda^4 }} = 2\pi\times ' if rrr > 300 else '')
                + f'{vortex_characteristic_freq_lambda/1e+9:.3g}$ GHz',
            xy=(vortex_characteristic_freq_lambda, 0.6),
            xycoords=plot_ax.get_xaxis_transform(),
            color=color,
            xytext=(+3, 0),
            textcoords='offset points',
            rotation=90,
            ha='left',
            va='top',
            fontsize='small',
        )
    if vspan is not None:
        plot_ax.axvspan(*vspan, **style.axvspan_kw)
    secax = plot_ax.secondary_yaxis('right', functions=(lambda x: geom_factor_f / x,)*2)
    if ax is None:
        plot_ax.set_ylabel(R'$R_\text{vortex}(\omega)/B$ ($\Omega$/G)')
        secax.set_ylabel('Limiting finesse at $B = 1$ G')
    plot_ax.set_xscale('log')
    plot_ax.set_yscale('log')
    # ax.legend(ncols=2)
    return plot_ax, secax


def bcs_subplot(
        temp_fit: TemperatureFit,
        roughness_rms: float,
        ax: Optional[Axes] = None,
        scale: Literal['linear', 'reci', 'reci_r'] = 'linear',
        minimal_limit_labels = False,
        offset_pts = (0, +3),
        **kwargs: ErrorbarKwargs,
    ):
    plot_ax: Axes
    if ax is None:
        _, plot_ax = plt.subplots()
    else:
        plot_ax = ax

    roughness_limit = roughness_limit_finesse(temp_fit.mode_frequency, roughness_rms)

    # plot the roughness one first to have it under
    temp_fit.plot_fit(roughness_limit, ax=plot_ax, color=style.limit_line_color, linestyle='dashed', scale=scale)
    temp_fit.plot(plot_limit_finesse=False, ax=plot_ax, **kwargs)
    annotate_line(
        ax,
        ('roughness' if minimal_limit_labels else 'roughness-limited'),
        temp_fit.plot_t_lims()[0],
        line_func=(lambda x: np.full_like(x, roughness_limit)),
        offset_pts=offset_pts,
        color=style.limit_line_color,
        horizontalalignment='left',
        verticalalignment='baseline',
        fontsize='small',
    )
