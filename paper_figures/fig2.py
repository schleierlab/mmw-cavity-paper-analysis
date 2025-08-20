import itertools
from collections.abc import Iterable
from typing import Final, Literal, assert_never, cast

import matplotlib.cm
import matplotlib.colors
import matplotlib.projections
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats
import skrf as rf
import uncertainties
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Affine2DBase, ScaledTranslation
from numpy.typing import NDArray
from tqdm import tqdm
from uncertainties import unumpy as unp

from suprtools.fp_theory.coupling_config import CouplingConfig
from suprtools.fp_theory.geometry import SymmetricCavityGeometry
from suprtools.fp_theory.geometry.anticrossing import AnticrossingFit
from suprtools.plotting import expand_range
from suprtools.plotting.units import Units
from suprtools.rf import WideScanNetwork
from suprtools.typing import ErrorbarKwargs, PlotKwargs

from . import style

theory_line_kw = PlotKwargs(
    color="0.6",
    alpha=0.5,
    # linestyle="dashed",
)


errorbar_kw = ErrorbarKwargs(
    linestyle="None",
    color="0.6",
    alpha=0.8,
    ecolor="0.5",
    capsize=1,
)


def plot_line_label(branch_sign, polarization):
    branch_label = ("upper" if branch_sign == +1 else "lower") + " branch"
    if polarization == "$x$":
        return f"{branch_label}, $x$ pol"
    elif polarization == "$y$":
        return "... $y$ polarized"


def make_avoided_crossing_plot(
    freq_ax: Axes,
    avoided_crossing_fit: AnticrossingFit,
    uparams,
    plot_inds=None,
    *,
    cmap=style.div_cmap,
    cmap_norm=style.mode_mixing_cmap_norm,
    theory_expansion_factor=1.15,
    frequency_subtraction: Literal['fsr', '00'] = 'fsr',
):
    if plot_inds is None:
        plot_inds = np.asarray(sorted(set(avoided_crossing_fit.modedata["q"])))
    
    fsr = avoided_crossing_fit.geometry.fsr
    if frequency_subtraction == 'fsr':
        freq_ax.set_ylabel(f"Frequency\nmodulo FSR ({Units.GHZ.mplstr()})")
        freq_yscale = 1e+9
        freq_ax.yaxis.set_major_formatter("{x:.3f}")
        def freq_subber(freq, q):
            return freq - fsr * q
    elif frequency_subtraction == '00':
        freq_ax.set_ylabel(Rf"$\nu - \nu_{{00}}$ ({Units.MHZ.mplstr()})")
        freq_yscale = 1e+6

        # TODO simplify this by making 00-mode-finding 
        def freq_subber(freq, q):
            if isinstance(q, Iterable):
                return np.asarray([
                    freq_subber(single_freq, single_q)
                    for (single_freq, single_q) in zip(freq, q)
                ])
            else:
                freq_00 = avoided_crossing_fit.geometry.near_confocal_coupling_matrix(q, CouplingConfig.no_xcoupling, max_order=0).eigvals.mean()
                return freq - freq_00
    else:
        assert_never(frequency_subtraction)

    for branch_ind in range(3):
        if branch_ind == 0:
            branch_sgn = +1
        elif branch_ind == 1:
            continue
        elif branch_ind == 2:
            branch_sgn = -1

        this_branch_qbranch = np.transpose(
            [plot_inds, np.full_like(plot_inds, branch_sgn)]
        )
        tem00_fracs = avoided_crossing_fit.mixing_fraction(this_branch_qbranch)

        for pol_ind, (polarization, marker) in enumerate([("$x$", "."), ("$y$", "x")]):

            ufreqs, ufwhms = np.array(uparams[pol_ind][branch_ind])[:].T
            freqs = unp.nominal_values(ufreqs)

            scatter_kw = dict(
                marker=marker,
                cmap=cmap,
                norm=cmap_norm,
            )

            qs = np.floor_divide(freqs, fsr)
            freq_ax.scatter(
                freqs / 1e9,
                freq_subber(freqs, qs) / freq_yscale,
                s=64,
                c=tem00_fracs,
                label=plot_line_label(branch_sgn, polarization),
                **scatter_kw,
            )

    # shape: n_qs, n_branches, 2
    theory_qs = np.linspace(*expand_range(plot_inds, theory_expansion_factor))
    qbranch_grid: NDArray = np.moveaxis(
        np.array([*np.meshgrid(theory_qs, [+1, -1])]),
        0,
        -1,
    )
    fit_freqs = avoided_crossing_fit.mixed_mode_frequency(qbranch_grid)
    for i, freq_set in enumerate(fit_freqs):
        freq_ax.plot(
            freq_set / 1e9,
            freq_subber(freq_set, theory_qs) / freq_yscale,
            # color=f'C{2*i}',
            # marker='^',
            **theory_line_kw,
        )

    fig = cast(Figure, freq_ax.get_figure())

    for i, q in enumerate([max(plot_inds), 26, min(plot_inds)]):
        # TODO unify this with code in NearConfocalCouplingMatrix.plot_mode_insets
        # inset_ax = inset_axes(
        #     freq_ax,
        #     0.25,
        #     0.25,
        #     bbox_to_anchor=(1 - 0.2 * i, 0, 0, 0),
        #     bbox_transform=(freq_ax.transAxes + offset_trans),
        #     loc='lower right',
        #     axes_class=matplotlib.projections.polar.PolarAxes,
        # )
        mode_freq_theory = avoided_crossing_fit.mixed_mode_frequency([(q, -1)]).item()
        offset_trans = ScaledTranslation(
            1 - 0.2 * (1 + i),
            0.05,
            cast(Affine2DBase, freq_ax.transAxes),
        )

        inset_ax = freq_ax.inset_axes(
            (0, 0, 0.35, 0.35),
            transform=(fig.dpi_scale_trans + offset_trans),
            # axes_class=matplotlib.projections.polar.PolarAxes,
        )
        freq_ax.indicate_inset(
            (
                mode_freq_theory / 1e+9,
                freq_subber(mode_freq_theory, q) / freq_yscale,
                0,
                0,
            ),
            inset_ax=inset_ax,
            edgecolor='0.5',
            alpha=0.2,
            # transform=freq_ax.transData,
        )
        avoided_crossing_fit.basis.plot_field_intensity(
            avoided_crossing_fit.mixed_mode_vector((q, -1)),
            projection='rectilinear',
            plot_range=4,
            ax=inset_ax,
            # cmap='magma',
            # norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e+0),
        )
        inset_ax.axis('off')
    
    freq_ax.set_xlabel(f"Frequency ({Units.GHZ.mplstr()})")


def mixed_mode_finesse_plot(
    fin_ax: Axes,
    avoided_crossing_fit: AnticrossingFit,
    uparams,  # 4D array indexed by polarization, branch, q, {freq, finesse}
    plot_inds=None,
    *,
    cmap=style.div_cmap,
    cmap_norm=style.mode_mixing_cmap_norm,
    theory_expansion_factor=1.15,
    fit: bool = True,
):
    if plot_inds is None:
        plot_inds = np.asarray(sorted(set(avoided_crossing_fit.modedata["q"])))
    for branch_ind in range(3):
        if branch_ind == 0:
            branch_sgn = +1
        elif branch_ind == 1:
            continue
        elif branch_ind == 2:
            branch_sgn = -1

        this_branch_qbranch = np.transpose(
            [plot_inds, np.full_like(plot_inds, branch_sgn)]
        )
        tem00_fracs = avoided_crossing_fit.mixing_fraction(this_branch_qbranch)

        for pol_ind, (polarization, marker) in enumerate([("$x$", "."), ("$y$", "x")]):
            ufreqs, ufwhms = np.array(uparams[pol_ind][branch_ind])[:].T
            ufins = avoided_crossing_fit.geometry.fsr / ufwhms
            freqs = unp.nominal_values(ufreqs)

            plotline, _, _ = fin_ax.errorbar(
                freqs / 1e9,
                unp.nominal_values(ufins),
                unp.std_devs(ufins),
                **errorbar_kw,
            )
            plotline.set_alpha(0.5)

            scatter_kw = dict(
                marker=marker,
                cmap=cmap,
                norm=cmap_norm,
            )

            fin_ax.scatter(
                freqs / 1e9,
                unp.nominal_values(ufins),
                s=None,
                c=tem00_fracs,
                **scatter_kw,
            )

    fin_fit_ufwhms = uparams[:, [0, 2], :][..., 1]
    branch_fit_shape = fin_fit_ufwhms.shape
    branch_shaped = np.broadcast_to(
        np.array([+1, -1]).reshape(1, -1, 1),
        branch_fit_shape,
    )
    qs_shaped = np.broadcast_to(plot_inds.reshape(1, 1, -1), branch_fit_shape)

    not_nan_mask = ~np.isnan(unp.nominal_values(fin_fit_ufwhms.flatten()))

    # fin_ax.set_yscale('log')
    fin_ax.set_ylabel("Finesse")
    fin_ax.set_xlabel(f"Frequency ({Units.GHZ.mplstr()})")

    if not fit:
        return

    wavg_finfit_upopt = mixing_wavg_fit(
        avoided_crossing_fit,
        qs_shaped.flatten()[not_nan_mask],
        branch_shaped.flatten()[not_nan_mask],
        fin_fit_ufwhms.flatten()[not_nan_mask],
        p0=(1e4, 1e6),
        bounds=(
            [1e3, 1e5],
            [1e5, 1e7],
        ),
    )

    # shape: n_qs, n_branches, 2
    theory_qs = np.linspace(*expand_range(plot_inds, theory_expansion_factor))
    qbranch_grid: NDArray = np.moveaxis(
        np.array([*np.meshgrid(theory_qs, [+1, -1])]),
        0,
        -1,
    )
    fit_freqs = avoided_crossing_fit.mixed_mode_frequency(qbranch_grid)
    for i, freq_set in enumerate(fit_freqs):
        branch_sgn = 1 - 2 * i
        this_branch_qbranch = np.transpose(
            [theory_qs, np.full_like(theory_qs, branch_sgn)]
        )
        fin_ax.plot(
            freq_set / 1e9,
            avoided_crossing_fit.geometry.fsr
            / mixed_mode_weighted_avg(
                avoided_crossing_fit,
                this_branch_qbranch,
                *unp.nominal_values(wavg_finfit_upopt),
            ),
            **theory_line_kw,
        )


def mixed_mode_residue_plot(
    res_ax: Axes,
    avoided_crossing_fit: AnticrossingFit,
    modeparams,
    uparams,
    insertion_loss_db,
    *,
    plot_inds=None,
    cmap=style.div_cmap,
    cmap_norm=style.mode_mixing_cmap_norm,
    theory_expansion_factor=1.15,
    fit: bool = True,
    short_labels: bool = False,
):
    if plot_inds is None:
        plot_inds = np.asarray(sorted(set(avoided_crossing_fit.modedata["q"])))

    coupling_rates: Final = modeparams.res_mag * rf.db_2_mag(insertion_loss_db)

    for branch_ind in range(3):
        if branch_ind == 0:
            branch_sgn = +1
        elif branch_ind == 1:
            continue
        elif branch_ind == 2:
            branch_sgn = -1

        this_branch_qbranch = np.transpose(
            [plot_inds, np.full_like(plot_inds, branch_sgn)]
        )
        tem00_fracs = avoided_crossing_fit.mixing_fraction(this_branch_qbranch)

        for pol_ind, (polarization, marker) in enumerate([("$x$", "."), ("$y$", "x")]):
            ufreqs, _ = np.array(uparams[pol_ind][branch_ind])[:].T
            freqs = unp.nominal_values(ufreqs)

            coupling_rates_thispol = coupling_rates[pol_ind, branch_ind]

            scatter_kw = dict(
                marker=marker,
                cmap=cmap,
                norm=cmap_norm,
            )
            plotline, _, _ = res_ax.errorbar(
                freqs / 1e9,
                unp.nominal_values(coupling_rates_thispol),
                unp.std_devs(coupling_rates_thispol),
                **errorbar_kw,
            )
            plotline.set_alpha(0.5)

            res_ax.scatter(
                freqs / 1e9,
                unp.nominal_values(coupling_rates_thispol),
                c=tem00_fracs,
                **scatter_kw,
            )

    if short_labels:
        res_ax.set_ylabel(fR"$\sqrt{{\kappa_1 \kappa_2}}/2\pi$ ({Units.HZ.mplstr()})")
    else:
        res_ax.set_ylabel("Probe coupling\n" Rf"$\sqrt{{\kappa_1 \kappa_2}}/2\pi$ ({Units.HZ.mplstr()})")
    res_ax.set_yscale("log")
    res_ax.set_xlabel(f"Frequency ({Units.GHZ.mplstr()})")

    if not fit:
        return

    fin_fit_ufwhms = uparams[:, [0, 2], :][..., 1]
    branch_fit_shape = fin_fit_ufwhms.shape
    branch_shaped = np.broadcast_to(
        np.array([+1, -1]).reshape(1, -1, 1),
        branch_fit_shape,
    )
    qs_shaped = np.broadcast_to(plot_inds.reshape(1, 1, -1), branch_fit_shape)

    coupling_fit_ucoupls = coupling_rates[:, [True, False, True], :]
    coupl_not_nan_mask = ~np.isnan(unp.nominal_values(coupling_fit_ucoupls.flatten()))
    wavg_coupling_fit_upopt = mixing_wavg_fit(
        avoided_crossing_fit,
        qs_shaped.flatten()[coupl_not_nan_mask],
        branch_shaped.flatten()[coupl_not_nan_mask],
        coupling_fit_ucoupls.flatten()[coupl_not_nan_mask],
        x_scale=[1, 1000],
        p0=(1, 1000),
        bounds=[
            (0.1, 100),
            (100, 1e5),
        ],
        log=True,
    )

    # shape: n_qs, n_branches, 2
    theory_qs = np.linspace(*expand_range(plot_inds, theory_expansion_factor))
    qbranch_grid: NDArray = np.moveaxis(
        np.array([*np.meshgrid(theory_qs, [+1, -1])]),
        0,
        -1,
    )
    fit_freqs = avoided_crossing_fit.mixed_mode_frequency(qbranch_grid)
    for i, freq_set in enumerate(fit_freqs):
        branch_sgn = 1 - 2 * i
        this_branch_qbranch = np.transpose(
            [theory_qs, np.full_like(theory_qs, branch_sgn)]
        )
        res_ax.plot(
            freq_set / 1e9,
            mixed_mode_weighted_avg(
                avoided_crossing_fit,
                this_branch_qbranch,
                *unp.nominal_values(wavg_coupling_fit_upopt),
            ),
            **theory_line_kw,
        )


def mixed_mode_weighted_avg(
    avoided_crossing_fit: AnticrossingFit,
    qbranch,
    val_00,
    val_n4,
):
    frac = avoided_crossing_fit.mixing_fraction(qbranch)
    return val_00 * frac + val_n4 * (1 - frac)


def mixing_wavg_fit(
    avoided_crossing_fit: AnticrossingFit,
    qs,
    branchs,
    uvals,
    *,
    log=False,
    plot=False,
    **kwargs,
):
    if plot:
        fig, ax = plt.subplots()
    fit_vals = unp.log10(uvals) if log else uvals

    with tqdm() as pbar:

        def weighted_avg(qbranch, val_00, val_n4):
            pbar.update(1)
            wavg = mixed_mode_weighted_avg(
                avoided_crossing_fit, qbranch, val_00, val_n4
            )
            retval = np.log10(wavg) if log else wavg

            rms_err = np.sqrt(np.mean((retval - unp.nominal_values(fit_vals)) ** 2))
            chi2 = np.sqrt(
                np.mean(
                    ((retval - unp.nominal_values(fit_vals)) / unp.std_devs(fit_vals))
                    ** 2
                )
            )
            pbar.set_description(
                f"tried values {val_00:.3g}, {val_n4:.3g}, rms err = {rms_err}, chi2 = {chi2}",
            )

            if plot:
                for branch in [+1, -1]:
                    mask = qbranch[:, 1] == branch
                    color = f"C{branch + 1}"
                    ax.plot(
                        qbranch[mask][:, 0],
                        unp.nominal_values(uvals[mask]),
                        marker=".",
                        linestyle="",
                        color=color,
                    )
                    ax.semilogy(
                        qbranch[mask][:, 0],
                        mixed_mode_weighted_avg(
                            avoided_crossing_fit, qbranch[mask], val_00, val_n4
                        ),
                        color=color,
                    )
                plt.pause(2)

            return retval

        popt, pcov = scipy.optimize.curve_fit(
            weighted_avg,
            xdata=np.asarray([qs, branchs]).T,
            ydata=unp.nominal_values(fit_vals),
            sigma=unp.std_devs(fit_vals),
            **kwargs,
        )

    return uncertainties.correlated_values(popt, pcov)


def make_avoided_crossing_plot_column(
    fig: Figure,
    axs: tuple[Axes, ...],
    avoided_crossing_fit: AnticrossingFit,
    modeparams,
    uparams,
    insertion_loss_db,
    *,
    cmap=style.div_cmap,
    cmap_norm=style.mode_mixing_cmap_norm,
    colorbar_kw=dict(),
    cbar_label_kw=dict(),
    theory_expansion_factor=1.15,
    wavg_fits: bool = True,
    short_labels=False,
):
    if len(axs) == 3:
        freq_ax, fin_ax, res_ax = axs
        plot_residues = True
    elif len(axs) == 2:
        freq_ax, fin_ax = axs
        plot_residues = False
    elif len(axs) == 4:
        cax, freq_ax, fin_ax, res_ax = axs
        plot_residues = True
    else:
        raise ValueError

    make_avoided_crossing_plot(
        freq_ax,
        avoided_crossing_fit,
        uparams,
        cmap=cmap,
        cmap_norm=cmap_norm,
        theory_expansion_factor=theory_expansion_factor,
        frequency_subtraction='00',
    )
    mixed_mode_finesse_plot(
        fin_ax,
        avoided_crossing_fit,
        uparams=uparams,
        cmap=cmap,
        cmap_norm=cmap_norm,
        fit=wavg_fits,
        theory_expansion_factor=theory_expansion_factor,
    )
    fin_ax.set_yscale("log")
    if plot_residues:
        mixed_mode_residue_plot(
            res_ax,
            avoided_crossing_fit,
            modeparams,
            uparams,
            insertion_loss_db,
            cmap=cmap,
            cmap_norm=cmap_norm,
            fit=wavg_fits,
            theory_expansion_factor=theory_expansion_factor,
            short_labels=short_labels,
        )

    for ax in axs:
        ax.set_xlabel("")

    axs[-1].set_xlabel(f"Frequency ({Units.GHZ.mplstr()})")

    cbar_loc_kw: dict  # declare type to avoid mypy anger in the `elif`
    if len(axs) < 4:
        cbar_loc_kw = dict(ax=axs)
    elif len(axs) == 4:
        cbar_loc_kw = dict(cax=cax)

    mappable = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap)
    cb = fig.colorbar(
        mappable,
        **cbar_loc_kw,
        **(style.colorbar_kw | colorbar_kw),
    )
    cb.set_label(
        R"$\mathrm{TEM}_{00}$ power fraction",
        **cbar_label_kw,
    )

    # fig.align_ylabels()
    return cb


def make_parax_shifts_column(
    fig,
    axs_all,
    start_freq_ghz,
    stop_freq_ghz,
    *,
    s21_network: WideScanNetwork,
    geo: SymmetricCavityGeometry,
    filt,
    max_order=4,
    mode_colorfunc=style.mode_colorfunc,
    annotate_kw=dict(),
    plot_kw=dict(),
    inset_kw=dict(),
    parax_inset_kw=dict(),
):
    """
    Make the right half of Figure 2.

    Parameters
    ----------
    axs_all:
    """
    axs_all = np.asarray(axs_all)
    axs_all[-1].set_xlabel(f"Frequency ({Units.GHZ.mplstr()})")

    # don't use Axes.remove() -- this causes an unclear bug in figure rendering for some installs
    axs_all[1].set_visible(False)

    axs = axs_all[2:]

    center_freq_ghz = (start_freq_ghz + stop_freq_ghz) / 2
    span_ghz = stop_freq_ghz - start_freq_ghz

    s21_network.fsr_compare_plots(
        center_freq_ghz,
        span_ghz,
        geo.fsr / 1e9,
        offsets=[0],
        filt=filt,
        fig=fig,
        axs=axs,
        rasterized=False,
        # geo=geo,  # TODO use `geo` kwarg and deprecate others
        **plot_kw,
    )

    ax_parax = axs_all[0]
    axs_all[2].set_ylabel(f"$|S_{{21}}|$ ({Units.DB.mplstr()})")
    axs_all[3].set_ylabel(R"$|S_{21, \text{HP}}|$")
    axs_all[3].ticklabel_format(useMathText=True)
    annotate_kw_default = dict(
        linestyle="dashed",
        alpha=0.35,
    )
    annotate_kwargs = {**annotate_kw_default, **annotate_kw}

    for axpair, offset_ind in zip(axs.reshape(-1, 2), itertools.count(0)):
        q_base = 26 + offset_ind

        diag_result = geo.near_confocal_coupling_matrix(
            q_base,
            CouplingConfig.no_xcoupling,
            max_order=max_order,
        )

        inds = slice(None, None, 2)
        offset = offset_ind * geo.fsr

        for ax in axpair:
            diag_result.annotate_modes(
                inds,
                offset=offset,
                scaling=1e9,
                ax=ax,
                label=False,
                color=mode_colorfunc,
                **annotate_kwargs,
            )
        inset_kw_default = dict(
            gap=24e6,
            inset_size=0.3,
            inset_pad=0.18,
            rasterized=True,
        )
        inset_kwargs = {**inset_kw_default, **inset_kw}
        diag_result.plot_mode_insets(
            inds,
            offset=offset,
            scaling=1e9,
            fig=fig,
            ax=axpair[0],
            **inset_kwargs,
        )

        diag_result_parax = geo.near_confocal_coupling_matrix(
            q_base,
            CouplingConfig.paraxial,
            max_order=max_order,
        )
        diag_result_parax.annotate_modes(
            inds,
            offset=offset,
            scaling=1e9,
            ax=ax_parax,
            label=0.8,
            color=mode_colorfunc,
            **annotate_kwargs,
        )
        parax_inset_kw_default = dict(
            gap=24e6,
            inset_size=0.3,
            inset_stagger=0.4,
            inset_pad=0.18,
            rasterized=True,
            projection="rectilinear",
        )
        parax_inset_kwargs = {**parax_inset_kw_default, **parax_inset_kw}
        diag_result_parax.plot_mode_insets(
            inds,
            offset=offset,
            scaling=1e9,
            fig=fig,
            ax=ax_parax,
            **parax_inset_kwargs,
        )

        for mode_order in range(0, max_order + 1, 2):
            parax_modes = sorted(
                [
                    eigval
                    for (eigval, eigvec) in zip(
                        diag_result_parax.eigvals, diag_result_parax.eigvecs
                    )
                    if diag_result_parax.basis[np.argmax(np.abs(eigvec) ** 2)].n
                    == mode_order
                ]
            )
            postparax_modes = sorted(
                [
                    eigval
                    for (eigval, eigvec) in zip(
                        diag_result.eigvals, diag_result.eigvecs
                    )
                    if diag_result.basis[np.argmax(np.abs(eigvec) ** 2)].n == mode_order
                ]
            )

            # only show every other mode, since they come in polarization doublets
            for par_val, post_val in zip(parax_modes[::2], postparax_modes[::2]):
                con = ConnectionPatch(
                    xyA=(par_val / 1e9, 0),
                    xyB=(post_val / 1e9, 1),
                    coordsA=ax_parax.get_xaxis_transform(),
                    coordsB=axpair[0].get_xaxis_transform(),
                    axesA=ax_parax,
                    axesB=axpair[0],
                    **{
                        **annotate_kwargs,
                        "color": mode_colorfunc(mode_order),
                    },
                )
                con.set_in_layout(False)
                ax_parax.add_artist(con)

    ax_parax.spines["right"].set_visible(False)
    ax_parax.spines["top"].set_visible(False)
    ax_parax.spines["left"].set_visible(False)
    ax_parax.spines["bottom"].set_visible(True)
    # ax_parax.yaxis.set_visible(False)
    ax_parax.yaxis.set_ticks([])
    ax_parax.set_ylabel("Paraxial")

    # fig.align_ylabels()
