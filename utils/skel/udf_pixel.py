from scipy import interpolate
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import warnings
import os
import traceback

try:
    import bottleneck as bn
except ImportError:
    bn = None

"""
>>> Mowing detection
>>> Copyright (C) 2021 Marcel Schwieder and Max Wesemeyer
"""


_warnings_configured = False
_cached_force_dates = None
_cached_force_year_fractions = None
_logged_exception_count = 0
_max_logged_exceptions = 20


def _write_zero_output(outarray):
    outarray[:] = 0


def _empty_detection_result():
    if profileAnalytics:
        return [], [], 0, 0, 0, 0, [], [], [], []
    return [], [], 0, 0, 0, 0


def _get_error_log_path():
    explicit_path = os.environ.get("MOWING_UDF_ERROR_LOG")
    if explicit_path:
        return explicit_path
    return os.path.join(os.path.dirname(__file__), "udf_pixel_errors.log")


def _log_forcepy_exception(exc, dates, nodata, ts):
    global _logged_exception_count

    if _logged_exception_count >= _max_logged_exceptions:
        return

    _logged_exception_count += 1

    try:
        finite_mask = np.isfinite(ts)
        finite_values = ts[finite_mask]
        min_value = float(np.min(finite_values)) if finite_values.size else None
        max_value = float(np.max(finite_values)) if finite_values.size else None
        log_lines = [
            "=" * 80,
            f"timestamp={datetime.now(timezone.utc).isoformat()}",
            f"exception_index={_logged_exception_count}",
            f"exception_type={type(exc).__name__}",
            f"exception_message={exc}",
            f"dates_count={len(dates)}",
            f"nodata={nodata}",
            f"ts_shape={ts.shape}",
            f"ts_dtype={ts.dtype}",
            f"ts_all_nodata={bool(np.all(ts == nodata))}",
            f"ts_all_zero={bool(np.all(ts == 0))}",
            f"ts_finite_min={min_value}",
            f"ts_finite_max={max_value}",
            "traceback:",
            traceback.format_exc(),
            "",
        ]
        with open(_get_error_log_path(), "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_lines))
    except Exception:
        pass


def _nanstd(values):
    if bn is not None:
        return bn.nanstd(values)
    return np.nanstd(values)


def _nanmean(values):
    if bn is not None:
        return bn.nanmean(values)
    return np.nanmean(values)


def _nanmedian(values):
    if bn is not None:
        return bn.nanmedian(values)
    return np.nanmedian(values)


def _nanmax(values):
    if bn is not None:
        return bn.nanmax(values)
    return np.nanmax(values)


def _ensure_cached_force_dates(dates):
    global _cached_force_dates, _cached_force_year_fractions

    if (
        _cached_force_dates is not None
        and _cached_force_year_fractions is not None
        and len(_cached_force_dates) == len(dates)
        and np.array_equal(_cached_force_dates, dates)
    ):
        return

    _cached_force_dates = np.array(dates, copy=True)
    date_objects = [serial_date_to_string(imgDate) for imgDate in dates]
    _cached_force_year_fractions = np.array(list(map(toYearFraction, date_objects)))


def get_cso(x, y, nodata=-9999, verbose=False, SoS=2018.2, EOS=2018.85):
    if len(x) == 0 or len(y) == 0:
        return 0, 0, 0
    # if no gap is found it will return 5 days as gap
    # in case the last potential observation misses the function calculates the gap to the EOS
    if np.all(y == nodata):
        nodata_ratio = 0
        return nodata_ratio, (x[-1] - x[0]) * 365, nodata
    nodata_mask = y == nodata
    nodata_sum = int(np.count_nonzero(nodata_mask))

    nodata_ratio = 1 - (nodata_sum / len(y))
    data_gap_dates_list = []
    nodata_indices = np.flatnonzero(nodata_mask)
    if nodata_indices.size:
        split_points = np.where(np.diff(nodata_indices) > 1)[0] + 1
        gap_groups = np.split(nodata_indices, split_points)
        for group in gap_groups:
            first_idx = int(group[0])
            last_idx = int(group[-1])
            if first_idx < 1:
                continue
            end_idx = last_idx + 1
            if end_idx < len(x):
                gap_days = (x[end_idx] - x[first_idx - 1]) * 365
                data_gap_dates_list.append(gap_days)
    #########################
    # calculating gap to EOS
    valid_indices = np.flatnonzero(~nodata_mask)
    if valid_indices.size == 0:
        return 0, 0, 0
    last_valid_idx = int(valid_indices[-1])
    gap_to_EOS = (EOS - x[last_valid_idx]) * 365
    data_gap_dates_list.append(gap_to_EOS)
    #########################
    # calculating gap to SOS
    first_valid_idx = int(valid_indices[0])
    gap_to_SOS = (x[first_valid_idx] - SoS) * 365
    data_gap_dates_list.append(gap_to_SOS)
    #########################
    if int(max(data_gap_dates_list)) == 0:
        data_gap_dates_list.append(5)
    if verbose:
        print(max(data_gap_dates_list), 'MAX GAP')
        print(x, y)
    return nodata_ratio, max(data_gap_dates_list), len(y) - nodata_sum


def toYearFraction(date):
    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def detectMow_S2_new(xs, ys, clearWd, yr, type='ConHull', nOrder=3, model='linear'):
    global _warnings_configured
    if not _warnings_configured:
        warnings.simplefilter('ignore')
        _warnings_configured = True
    another_thrs = 0.15

    Y = np.asarray(ys) / 10000
    X = np.asarray(xs)
    clearWd_frac = clearWd * 0.00273973

    Season_min_frac = yr + GLstart
    Season_max_frac = yr + GLend
    Start_frac = yr + PSstart
    End_frac = yr + PSend

    if type == 'ConHull':
        validIndex = Y < 1
        Y = Y[validIndex]
        X = X[validIndex]
        validIndex_2 = Y > 0
        Y = Y[validIndex_2]
        X = X[validIndex_2]
        if Y.size == 0 or X.size == 0:
            return _empty_detection_result()

        ##############################################
        # averages duplicates in the time series
        records_array = X
        vals, inverse, count = np.unique(records_array, return_inverse=True, return_counts=True)
        Y = np.bincount(inverse, weights=Y) / count
        X = vals
        if Y.size == 0 or X.size == 0:
            return _empty_detection_result()

        ##############################################

        # filter time series to season (check if needed or a code legacy)
        SoGLS = np.abs(X - Season_min_frac).argmin()
        EoGLS = np.abs(X - Season_max_frac).argmin()
        Y = np.asarray(Y[SoGLS:EoGLS])
        X = np.asarray(X[SoGLS:EoGLS])
        if Y.size == 0 or X.size == 0 or not np.any(np.isfinite(Y)):
            return _empty_detection_result()

        # calculate NDVI difference (t1) - (t-1)
        yT1 = np.asarray(Y[1:])
        yT2 = np.asarray(Y[:-1])

        YDiffzero = [0]
        YDiff = yT1 - yT2
        YDiff = np.append(YDiffzero, YDiff)

        EVI_STD = _nanstd(Y)
        EVI_obs = sum(~np.isnan(Y))
        EVI_obs_pot = EVI_obs / len(Y)

        LoS = int(X[len(X) - 1] * 365 - X[0] * 365)
        if LoS <= 0:
            return _empty_detection_result()
        EVI_obs_potII = EVI_obs / (LoS / 5)

        # identify first peak somewhere around the "mid" of the season
        # DOY 120
        MoSStart = np.abs(X - Start_frac).argmin()

        # DOY 240
        MoSEnd = np.abs(X - End_frac).argmin()

        YPeakSub = Y[MoSStart:MoSEnd]

        if len(YPeakSub) == 0 or not np.any(np.isfinite(YPeakSub)):
            return _empty_detection_result()

        MoSPeak = _nanmax(YPeakSub)
        MoSIndex = int(np.nanargmax(YPeakSub)) + MoSStart

        earlyIndex2 = -1
        lateIndex2 = -1

        # todo check if early and late peak equals Y0
        Y0 = int(np.flatnonzero(np.isfinite(Y))[0])

        if MoSIndex <= 2:
            if MoSIndex == 0:
                earlyPeak1 = Y[0]
            else:
                earlyPeak1 = _nanmax(Y[0:MoSIndex])
            earlyIndex1 = int(np.flatnonzero(Y == earlyPeak1)[0])
        else:
            searchInd = np.flatnonzero(X <= X[MoSIndex] - clearWd_frac)
            if searchInd.size:
                searchInd = int(searchInd[-1])
                earlyPeak1 = _nanmax(Y[0:searchInd])
                earlyIndex1 = int(np.flatnonzero(Y == earlyPeak1)[0])
            else:
                earlyIndex1 = 0

        if MoSIndex + 2 == len(X):
            latePeak1 = _nanmax(Y[MoSIndex + 1:len(X)])
            lateIndex1 = int(np.flatnonzero(Y == latePeak1)[-1])
        else:
            searchInd2 = np.flatnonzero(X >= X[MoSIndex] + clearWd_frac)
            if searchInd2.size:
                searchInd2 = int(searchInd2[0])
                if searchInd2 != len(X) - 1:
                    latePeak1 = _nanmax(Y[searchInd2:len(X) - 1])
                    lateIndex1 = int(np.flatnonzero(Y == latePeak1)[-1])
                else:
                    lateIndex1 = 0
            else:
                lateIndex1 = 0

        if (earlyIndex1 != 0) and (earlyIndex1 - 2) > 0 and np.any(Y[0:earlyIndex1 - 2]):
            searchInd3 = np.flatnonzero(X <= X[earlyIndex1] - clearWd_frac)
            if searchInd3.size:
                searchInd3 = int(searchInd3[-1])
                earlyPeak2 = _nanmax(Y[0:searchInd3])
                earlyIndex2 = int(np.flatnonzero(Y == earlyPeak2)[0])

        if (lateIndex1 != 0) and lateIndex1 + 2 <= len(X) and np.any(Y[lateIndex1 + 2:len(X)]):
            searchInd4 = np.flatnonzero(X >= X[lateIndex1] + clearWd_frac)
            if searchInd4.size:
                searchInd4 = int(searchInd4[0])
                latePeak2 = _nanmax(Y[searchInd4:len(X)])
                lateIndex2 = int(np.flatnonzero(Y == latePeak2)[-1])

        xarr_indices = [Y0]
        if earlyIndex2 != -1:
            xarr_indices.append(earlyIndex2)
        xarr_indices.extend([earlyIndex1, MoSIndex, lateIndex1])
        if lateIndex2 != -1:
            xarr_indices.append(lateIndex2)
        xarr_indices.append(len(X) - 1)

        Xarr = [X[idx] for idx in xarr_indices]
        Yarr = [Y[idx] for idx in xarr_indices]
        if len(Xarr) < 2 or len(Yarr) < 2:
            return _empty_detection_result()

    if model == 'linear':
        # model and fit spline
        polyVal = np.interp(X, xp=Xarr, fp=Yarr)

    if model == 'poly':
        # model and fit polynom of n-th order
        poly = np.polyfit(Xarr, Yarr, nOrder)
        polyVal = np.polyval(poly, X)

    if model == 'spline':
        tck = interpolate.splrep(x=Xarr, y=Yarr, s=0)

        #  predict values with spline and write to array
        polyVal = interpolate.splev(X, tck, der=0)

    # difference between polynom and values
    diff = np.abs(polyVal - Y)
    diff_sum = np.nansum(diff)
    diff_mean = _nanmean(diff)
    testVal = diff_sum * EVI_obs_potII

    thresh = diff_mean
    NDVIthresh = -EVI_STD
    NDVIthresh_list = list(np.random.normal(NDVIthresh, GFstd, 100))

    # create empty array for neighborhood indices
    clearWidth = []

    mow_date_index = []
    mowingEvents = []
    mowingDoy = []

    if len(diff) > 0:
        i = 1
        for evIndex, ev in enumerate(diff):
            ndvi_diff_check = False
            if np.count_nonzero(YDiff[evIndex] < NDVIthresh_list) >= posEval:
                ndvi_diff_check = True
            else:
                continue

            eventDate = X[evIndex]

            if evIndex == len(X) - 1:
                eventDate_next = X[evIndex] + 1
            else:
                eventDate_next = X[evIndex + 1]

            if i == 1:
                if ev > thresh:
                    # check NDVI difference and compare to threshold
                    if ndvi_diff_check:
                        # check next observation
                        if eventDate_next - eventDate <= 6 * 0.00273973:
                            if YDiff[evIndex + 1] > another_thrs:
                                continue
                        # get julian date
                        doy = ((eventDate - yr) * 365) + 1
                        if doy > 305:
                            continue
                        else:
                            dt = datetime(yr, 1, 1)
                            dtdelta = timedelta(days=doy)
                            dates = str(dt + dtdelta)
                            date = dates[0:10]
                            mowingEvents.append(date)
                            mowingDoy.append(int(doy))
                            mow_date_index.append(evIndex)
                            i = i + 1
            else:
                if ev > thresh:
                    dec_date_preceding = X[np.array(mow_date_index)[-1]]
                    dec_date_current_iter = X[evIndex]
                    # delta days in decimal format
                    delta_days = dec_date_current_iter - dec_date_preceding
                    # clearwd (days) divided by 365 = minimum distance from preceding mowing event as decimal number
                    clearWd_days = clearWd / 365
                    if delta_days > clearWd_days:
                        # if evIndex not in clearWidth:
                        # date of event when threshold was crossed
                        eventDate = X[evIndex]
                        if ndvi_diff_check:
                            if eventDate_next - eventDate <= 6 * 0.00273973:
                                if YDiff[evIndex + 1] > another_thrs:
                                    continue
                            # get julian date
                            doy = ((eventDate - yr) * 365) + 1
                            if doy > 305:
                                continue
                            else:
                                #############################
                                # check if there is one observation that is higher than the preceding between
                                # two mowing events
                                time_mask = np.where((X >= X[mow_date_index[-1]]) & (X <= eventDate), True, False)
                                any_preced_lower = np.any(np.ediff1d(Y[time_mask]) > 0)
                                #############################
                                if any_preced_lower:
                                    dt = datetime(yr, 1, 1)
                                    dtdelta = timedelta(days=doy)
                                    dates = str(dt + dtdelta)
                                    date = dates[0:10]
                                    mowingEvents.append(date)
                                    mowingDoy.append(int(doy))
                                    mow_date_index.append(evIndex)
                                    i = i + 1
                    else:
                        None

    if profileAnalytics:
        return mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, testVal, Xarr, Yarr, X, polyVal
    else:
        return mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, testVal


# new version
def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """
    _ensure_cached_force_dates(dates)

    bandnames = ['mowingEvents', 'max_gap_days', 'CSO_ABS', 'Data_Ratio',
                 'Mow_1', 'Mow_2', 'Mow_3', 'Mow_4', 'Mow_5', 'Mow_6', 'Mow_7', 'Mean', 'Median', 'SD', 'diff_sum',
                 'diff_sum_dataavail', 'Error']

    return bandnames


def serial_date_to_string(srl_no):
    # FORCE dates are days since 1970-01-01. Do not shift by -1 day,
    # otherwise Jan 1 observations get pulled into the previous year.
    new_date = datetime(1970, 1, 1, 0, 0) + timedelta(days=int(srl_no))
    return new_date


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """
    inarray:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16), nrows & ncols always 1
    outarray:  numpy.ndarray[nOutBands](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads (always 1)
    Write results into outarray.
    """
    global GLstart, GLend, GLendII, PSstart, PSend, GFstd, posEval, clrwd, profileAnalytics
    global _cached_force_dates, _cached_force_year_fractions

    profileAnalytics = False

    GLstart = 0.2  # DOY 73
    GLend = 1  # DOY 365
    GLendII = 0.85  # DOY
    PSstart = 0.33  # DOY 120
    PSend = 0.66  # DOY 240
    GFstd = 0.02
    posEval = 40
    clrwd = 15

    np.seterr(all='ignore')
    ts = inarray.squeeze()

    nodata = nodata

    all_no_data = np.all(ts == nodata)
    all_zero = np.all(ts == 0)

    if all_no_data:
        _write_zero_output(outarray)
        return
    elif all_zero:
        _write_zero_output(outarray)
        return
    else:

        try:
            if profileAnalytics:
                x = np.array(dates)
            else:
                _ensure_cached_force_dates(dates)
                x = _cached_force_year_fractions
                if x is None or len(x) == 0:
                    _write_zero_output(outarray)
                    return

            # Use the median year of the time series instead of the first
            # observation so year-boundary acquisitions do not anchor the
            # season to the previous year.
            yr = int(np.floor(np.nanmedian(x)))
            #################################
            # get sd mean median
            Season_min_frac = yr + GLstart
            Season_max_frac = yr + GLendII
            subsetter = np.where((Season_min_frac < x) & (x < Season_max_frac), True, False)
            if not np.any(subsetter):
                _write_zero_output(outarray)
                return

            Y = np.array(ts[subsetter])
            X = x[subsetter]
            if X.size == 0 or Y.size == 0:
                _write_zero_output(outarray)
                return
            nodata_ratio, max_gap_days, cso_abs = get_cso(X, Y, nodata=nodata, verbose=False, SoS=Season_min_frac,
                                                          EOS=Season_max_frac)
            Y = np.array(ts[subsetter], dtype=float)
            Y[Y == nodata] = np.nan
            if not np.any(np.isfinite(Y)):
                _write_zero_output(outarray)
                return
            mean = _nanmean(Y)
            median = _nanmedian(Y)
            sd = _nanstd(Y)

            Season_min_frac = yr + GLstart
            Season_max_frac = yr + GLend
            subsetter = np.where((Season_min_frac < x) & (x < Season_max_frac), True, False)
            if not np.any(subsetter):
                _write_zero_output(outarray)
                return
            X = x[subsetter]
            Y = ts[subsetter]
            if X.size == 0 or Y.size == 0:
                _write_zero_output(outarray)
                return

            if profileAnalytics:
                result = detectMow_S2_new(
                    X, Y, clearWd=clrwd, yr=yr, type='ConHull', nOrder=3, model='linear'
                )
                if result is None:
                    _write_zero_output(outarray)
                    return
                mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, diff_sum_dataavail, xPeak, yPeak, xPol, yPol = result
            else:
                result = detectMow_S2_new(
                    X, Y, clearWd=clrwd, yr=yr, type='ConHull', nOrder=3, model='linear'
                )
                if result is None:
                    _write_zero_output(outarray)
                    return
                mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, diff_sum_dataavail = result

            mowing_doy_out = [0] * 7

            for index, doys in enumerate(mowing_doy_out):
                try:
                    mowing_doy_out[index] = mowingDoy[index]
                except:
                    break
            outarray[:] = [int(len(mowingEvents)), int(max_gap_days), int(cso_abs), int(nodata_ratio * 100),
                           mowing_doy_out[0],
                           mowing_doy_out[1], mowing_doy_out[2], mowing_doy_out[3], mowing_doy_out[4],
                           mowing_doy_out[5], mowing_doy_out[6], mean, median, sd,
                           int(diff_sum * 100),
                           int(diff_sum_dataavail * 100), 0]
            if profileAnalytics:
                return mowingEvents, mowing_doy_out, xPeak, yPeak, xPol, yPol
        except Exception as exc:
            _log_forcepy_exception(exc, dates, nodata, ts)
            _write_zero_output(outarray)
