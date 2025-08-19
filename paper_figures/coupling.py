from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import skrf as rf
from numpy.typing import ArrayLike, NDArray
from suprtools.fp_theory.geometry import FiniteMirrorSymmetricCavity
from suprtools.gmsh_utils import CurlGradField
from suprtools.plotting import mpl_usetex
from suprtools.plotting.style import kwarg_func_factory
from suprtools.plotting.units import Units
from suprtools.rf import FabryPerotModeParams
from suprtools.rf.insertion_loss import LossElement
from suprtools.typing import PlotKwargs
from uncertainties import ufloat
from uncertainties import unumpy as unp

from . import constants, style


# TODO this needs a LOT of refactoring :<
def coupling_subfig(
        ax,
        probe,
        near_confocal_data: dict[tuple[int, int], tuple[float, ArrayLike, ArrayLike]],
        flaminia_cav: FiniteMirrorSymmetricCavity,
        flaminia_data: FabryPerotModeParams,
        flaminia_insertion_loss: LossElement,
        flaminia_sim_field: CurlGradField,
        flaminia_q_range: tuple[int, int],
        cut_field_r: NDArray,
        legend_kw: Optional[dict] = dict(),
        # errorbar_kw=dict(),
        flaminia_kwarg_func: Optional[Callable[[Any, int, int], PlotKwargs]] = None,
        stage_position_converter: Callable[[ArrayLike], NDArray] = constants.stage_position_to_radius,
        short_labels: bool = False,
):
    pol_markers = ('.', 'x')
    # pol_markers_alt = ('p', '^')
    pol_markers_alt = pol_markers
    
    theory_line_kw = dict(
        linestyle='dashed',
        label='Simulation estimate',
        marker=None,
    )

    ax.errorbar([], [], marker='', linestyle='', label='Near-confocal')
    
    for (q, pol), (freq, stage_pos, res_mags) in near_confocal_data.items():
        print(f'{freq=}')
        pol_str = 'x' if pol == +1 else 'y'
        probe_pos = stage_position_converter(stage_pos)
        geomean_couplings = rf.db_2_mag(constants.caecilia_cassia_insertion_loss_db) * np.array(res_mags)

        if mpl_usetex():
            label = fr'$\mathrm{{TEM}}_{{{q},0,0,{pol_str}}}$ @ \SI{{{freq/1e+9:.4f}}}{{\GHz}}'
        else:
            label = fr'$\mathrm{{TEM}}_{{{q},0,0,{pol_str}}}$ @ {freq/1e+9:.4f} GHz'

        ax.errorbar(
            unp.nominal_values(probe_pos),
            unp.nominal_values(geomean_couplings),
            xerr=unp.std_devs(probe_pos),
            yerr=unp.std_devs(geomean_couplings),
            marker=pol_markers_alt[int((1 - pol)//2)],
            markersize=4,
            color=style.near_confocal_color(q),
            capsize=2,
            elinewidth=1.5,
            linestyle='None',
            label=label,
        )
    
    # ax.errorbar(
    #     cut_field_r * 1e+3,
    #     field_to_coupling_rate(cassia_cut_field_normalized, 90.8182e+9),
    #     **(kwarg_func_factory(label=False, palette=alt_palette)(None, 26 + 1, -1) | theory_line_kw | dict(alpha=1)),
    # )
    # ax.errorbar([], [], marker='', linestyle='', label='')
    # ax.errorbar([], [], marker='', linestyle='', label='Less-confocal')
    
    print(f'{np.nanmean(unp.nominal_values(flaminia_data.freqs))=}')

    default_kwarg_func = kwarg_func_factory(
        label=True,
        q_range=flaminia_q_range,
        markers=pol_markers,
        linestyle='',
        # **errorbar_kw,
    )

    the_flaminia_kwargfunc = (
        default_kwarg_func
        if flaminia_kwarg_func is None else
        flaminia_kwarg_func
    )
    flaminia_data.errorbar_plot(
        (flaminia_data.res_mag * constants.distortion_uncertainty) / rf.db_2_mag(flaminia_insertion_loss.loss_db(unp.nominal_values(flaminia_data.freqs))),
        axis=0,
        ax=ax,
        axes_order=[1, 2],
        remove_nans=True,
        error_style='fill',
        kwarg_func=the_flaminia_kwargfunc,
    )
    
    def plot_uplims(stage_positions, estimated_residue, freq, **kwargs):
        fudge_factor = ufloat(1, 0.5)
        lims = fudge_factor * constants.distortion_uncertainty * estimated_residue / rf.db_2_mag(flaminia_insertion_loss.loss_db(freq))
        ax.errorbar(
            stage_position_converter(stage_positions),
            unp.nominal_values(lims),
            yerr=unp.std_devs(lims),
            **(kwargs | dict(uplims=True, linestyle='')),
        )
    
    # manually determined bounds for the residue for datasets with insufficient SNR for fitting
    # plot_kwarg_func_nolabel = kwarg_func_factory(label=False, q_range=flaminia_q_range, markers=pol_markers)
    plot_uplims(
        [-2.25, -3, -3.75],
        np.array([20, 8, 10]) * 1e-3,
        90.6544e+9,
        **(the_flaminia_kwargfunc(None, 27, +1) | dict(label=None)),
    )
    plot_uplims(
        [-2.25, -3, -3.75],
        np.array([24, 11, 16]) * 1e-3,
        90.6548e+9,
        **(the_flaminia_kwargfunc(None, 27, -1) | dict(label=None)),
    )

    flaminia_probe_z = flaminia_cav.offset_z_from_edge(constants.probe_offset)
    flaminia_cut_field_coords = cut_field_r[:, np.newaxis] * [1, 0, 0] + [0, flaminia_probe_z, 0]
    flaminia_cut_field_normalized = flaminia_sim_field.eval_field(flaminia_cut_field_coords)
    ax.errorbar(
        cut_field_r * 1e+3,
        probe.resonator_coupling_rate(flaminia_cut_field_normalized, 90.6546e+9),
        **(the_flaminia_kwargfunc(None, 27, -1) | theory_line_kw),
    )
    
    ax.set_yscale('log')

    if short_labels:
        ylabel_body = R'$\sqrt{\kappa_1\kappa_2}/2\pi$'
    else:
        ylabel_body = R'Probe coupling $\sqrt{\kappa_1\kappa_2}/2\pi$'
    ax.set_ylabel(f'{ylabel_body} ({Units.HZ.mplstr()})')

    # raw_handles, raw_labels = ax.get_legend_handles_labels()
    # empty_handles = [plt.plot([], marker="", ls="")[0]] * 2

    # n_flaminia_handles = np.prod(flaminia_sweep_subset.params_arr.shape[1:]) + 1
    # handles = empty_handles[:1] + raw_handles[:n_flaminia_handles] + empty_handles[1:] + raw_handles[n_flaminia_handles:]
    # labels = ['Near-confocal'] + raw_labels[:n_flaminia_handles] + ['Less-confocal'] + raw_labels[n_flaminia_handles:]

    if legend_kw is not None:
        legend_kwargs = dict(fontsize='x-small', ncol=1) | legend_kw
        ax.legend(
            # handles,
            # labels,
            # loc='center left', bbox_to_anchor=(1.005, 0.5),
            **legend_kwargs,
        )
