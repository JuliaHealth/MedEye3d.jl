import re
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pydicom
import SimpleITK as sitk

try:
    from ..exceptions import DataStructureError, DataStructureWarning
except (ImportError, ValueError):
    class DataStructureError(Exception):
        pass

    class DataStructureWarning(UserWarning):
        pass

ENHANCED_PET_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.130",  # Enhanced PET Image Storage
    "1.2.840.10008.5.1.4.1.1.128.1",  # Legacy Converted Enhanced PET Image Storage
}
LEGACY_CONVERTED_ENHANCED_PET_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.128.1"


def is_enhanced_pet(ds):
    """Return whether *ds* uses an Enhanced PET storage class."""
    return str(getattr(ds, "SOPClassUID", "")) in ENHANCED_PET_SOP_CLASS_UIDS


def _is_legacy_converted_enhanced_pet(ds):
    return str(getattr(ds, "SOPClassUID", "")) == LEGACY_CONVERTED_ENHANCED_PET_SOP_CLASS_UID


def is_fdg(name):
    fdg_pattern = re.compile(
        r"(fdg|fluorodeoxy|fludeoxy|2[-\s]?\[?18f\]?[-\s]?fluoro)",
        re.IGNORECASE,
    )
    return bool(fdg_pattern.search(name))


def _finite_positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DataStructureError(f"{name} is missing or invalid.")
    if not np.isfinite(number) or number <= 0:
        raise DataStructureError(f"{name} must be finite and > 0.")
    return number


def _finite_nonnegative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DataStructureError(f"{name} is missing or invalid.")
    if not np.isfinite(number) or number < 0:
        raise DataStructureError(f"{name} must be finite and >= 0.")
    return number


def _single_radiopharmaceutical_item(ds):
    sequence = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if sequence is None or len(sequence) == 0:
        raise DataStructureError("Radiopharmaceutical Information Sequence (0054,0016) is missing or empty.")
    if len(sequence) != 1:
        raise DataStructureError(
            "Multiple Radiopharmaceutical Information Sequence (0054,0016) items are not supported because "
            "the applicable administration cannot be identified."
        )
    return sequence[0]


def parse_time(time_str, vr=None):
    """Parse a DICOM DA, TM, or DT value into a datetime object."""
    if isinstance(time_str, bytes):
        time_str = time_str.decode("utf-8").strip()

    time_str = str(time_str).strip()
    match = re.fullmatch(r"(?P<components>\d+)(?P<fraction>\.\d{1,6})?(?P<offset>[+-]\d{4})?", time_str)
    if match is None:
        raise ValueError(f"Time data '{time_str}' does not match a DICOM date or time format")

    components = match.group("components")
    fraction = match.group("fraction") or ""
    offset = match.group("offset") or ""
    vr = vr.upper() if vr is not None else None
    if vr is None:
        if offset or len(components) > 8:
            vr = "DT"
        elif len(components) == 8:
            vr = "DA"
        else:
            vr = "TM"

    formats = {
        "DA": {8: "%Y%m%d"},
        "TM": {2: "%H", 4: "%H%M", 6: "%H%M%S"},
        "DT": {
            4: "%Y",
            6: "%Y%m",
            8: "%Y%m%d",
            10: "%Y%m%d%H",
            12: "%Y%m%d%H%M",
            14: "%Y%m%d%H%M%S",
        },
    }
    if vr not in formats or len(components) not in formats[vr]:
        raise ValueError(f"Time data '{time_str}' has invalid component width for DICOM {vr or 'value'}")
    if fraction and not ((vr == "TM" and len(components) == 6) or (vr == "DT" and len(components) == 14)):
        raise ValueError(f"Time data '{time_str}' has fractional seconds without a seconds component")
    if offset and vr != "DT":
        raise ValueError(f"Time data '{time_str}' has a UTC offset but is not a DICOM DT value")
    if offset:
        _parse_utc_offset(offset)

    fmt = formats[vr][len(components)]
    if fraction:
        fmt += ".%f"
    if offset:
        fmt += "%z"
    return datetime.strptime(components + fraction + offset, fmt)


def _parse_utc_offset(value):
    match = re.fullmatch(r"(?P<sign>[+-])(?P<hours>\d{2})(?P<minutes>\d{2})", str(value).strip())
    if match is None or str(value).strip() == "-0000":
        raise ValueError("Invalid DICOM UTC offset")

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    total_minutes = hours * 60 + minutes
    if (
        minutes >= 60
        or (match.group("sign") == "+" and total_minutes > 14 * 60)
        or (match.group("sign") == "-" and total_minutes > 12 * 60)
    ):
        raise ValueError("Invalid DICOM UTC offset")
    if match.group("sign") == "-":
        total_minutes = -total_minutes
    return timezone(timedelta(minutes=total_minutes))


def _time_component_width(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8").strip()
    match = re.match(r"\d+", str(value).strip())
    return len(match.group()) if match is not None else 0


def _dataset_timezone(ds):
    dataset_offset = getattr(ds, "TimezoneOffsetFromUTC", None)
    if dataset_offset in [None, ""]:
        return None
    try:
        return _parse_utc_offset(dataset_offset)
    except ValueError:
        raise DataStructureError("Timezone Offset From UTC (0008,0201) is invalid.")


def _datetime_on_reference_date(value, reference, infer_previous_day=True):
    # Vendor-private reference fields may contain either TM or a complete DT.
    raw_value = value.decode("utf-8").strip() if isinstance(value, bytes) else str(value).strip()
    components = re.match(r"\d+", raw_value)
    if components is not None and len(components.group()) == 14:
        parsed = parse_time(value, "DT")
        tzinfo = parsed.tzinfo if parsed.tzinfo is not None else reference.tzinfo
        return parsed.replace(tzinfo=tzinfo)

    parsed = parse_time(value, "TM")
    tzinfo = reference.tzinfo
    result = parsed.replace(
        year=reference.year,
        month=reference.month,
        day=reference.day,
        tzinfo=tzinfo,
    )
    if infer_previous_day and result > reference:
        result -= timedelta(days=1)
    return result


def get_radionuclide_half_life(ds):
    """Return radionuclide half-life in seconds."""
    rph = _single_radiopharmaceutical_item(ds)
    return _finite_positive(
        getattr(rph, "RadionuclideHalfLife", None),
        "Radionuclide Half Life (0018,1075)",
    )


def get_decay_correction_reference_datetime(ds, acquisition_time, decay_constant):
    """
    Return the datetime to which the PET activity is decay-corrected.

    This mirrors the existing vendor-specific START decay-correction logic and
    is used only to determine whether RadiopharmaceuticalStartDateTime has a
    plausible date offset.
    """
    if ds.DecayCorrection != "START":
        return acquisition_time

    manufacturer = ds.Manufacturer.upper()

    series_time = None
    try:
        series_date_value = getattr(ds, "SeriesDate", None)
        if series_date_value not in [None, ""]:
            series_date = parse_time(series_date_value, "DA")
            series_time = _datetime_on_reference_date(ds.SeriesTime, series_date, infer_previous_day=False)
        else:
            series_time = _datetime_on_reference_date(ds.SeriesTime, acquisition_time)
        if series_time.tzinfo is None and acquisition_time.tzinfo is not None:
            series_time = series_time.replace(tzinfo=acquisition_time.tzinfo)
    except (AttributeError, ValueError, TypeError):
        pass

    if series_time is not None and acquisition_time == series_time:
        return acquisition_time

    # Preserve the vendor-specific START strategies in one shared resolver so
    # validation and SUV application cannot choose different timestamps.
    if "SIEMENS" in manufacturer or "CPS" in manufacturer or "CTI" in manufacturer:
        try:
            return _datetime_on_reference_date(ds[(0x0071, 0x1022)].value, acquisition_time)
        except (KeyError, TypeError, ValueError):
            pass

    if "GE" in manufacturer:
        try:
            return _datetime_on_reference_date(ds[(0x0009, 0x100D)].value, acquisition_time)
        except (KeyError, TypeError, ValueError):
            frame_reference_time = (
                _finite_nonnegative(
                    ds.FrameReferenceTime,
                    "Frame Reference Time (0054,1300)",
                )
                / 1000.0
            )
            return acquisition_time - timedelta(seconds=frame_reference_time)

    frame_reference_time = (
        _finite_nonnegative(
            ds.FrameReferenceTime,
            "Frame Reference Time (0054,1300)",
        )
        / 1000.0
    )
    frame_duration = _finite_positive(
        ds.ActualFrameDuration,
        "Actual Frame Duration (0018,1242)",
    )
    decay_during_frame = decay_constant * frame_duration / 1000.0
    avg_count_rate_time = (1 / decay_constant) * np.log(decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return acquisition_time + timedelta(seconds=avg_count_rate_time - frame_reference_time)


def calc_start_elapsed_time(ds, decay_constant, acquisition_time, injection_time):
    """Return elapsed time using the shared vendor-specific START strategy."""
    decay_reference_time = get_decay_correction_reference_datetime(
        ds,
        acquisition_time,
        decay_constant,
    )
    return (decay_reference_time - injection_time).total_seconds()


def resolve_injection_and_acquisition_times(ds, half_life, decay_constant):
    """
    Resolve injection and acquisition datetimes.

    A complete RadiopharmaceuticalStartDateTime is trusted only when its offset
    to the decay-correction reference datetime is >= -1 hour and < 2 half-lives.
    For uncorrected frames, the reference is the acquisition datetime itself,
    so the offset must be nonnegative.

    For short-lived radionuclides, unreliable or missing dates can be replaced
    by a time-based inference. If injection time is later than acquisition time,
    injection is assumed to have occurred on the previous day, but only when
    the resulting difference is < 6 hours.

    For long-lived radionuclides, dates must be trustworthy because uptake may
    legitimately span more than one day.
    """
    rph = _single_radiopharmaceutical_item(ds)

    injection_datetime_value = getattr(rph, "RadiopharmaceuticalStartDateTime", None)
    injection_time_value = getattr(rph, "RadiopharmaceuticalStartTime", None)
    acquisition_date_value = getattr(ds, "AcquisitionDate", None)

    injection_datetime_present = injection_datetime_value not in [None, ""]
    injection_time_present = injection_time_value not in [None, ""]
    acquisition_date_present = acquisition_date_value not in [None, ""]

    if not injection_datetime_present and not injection_time_present:
        raise DataStructureError(
            "Both Radiopharmaceutical Start DateTime (0018,1078) and "
            "Radiopharmaceutical Start Time (0018,1072) are missing."
        )

    injection_datetime = None
    if injection_datetime_present:
        try:
            injection_datetime = parse_time(injection_datetime_value, "DT")
        except (ValueError, TypeError):
            raise DataStructureError("Radiopharmaceutical Start DateTime (0018,1078) is invalid.")

        datetime_component_width = _time_component_width(injection_datetime_value)
        if datetime_component_width < 8:
            raise DataStructureError("Radiopharmaceutical Start DateTime (0018,1078) does not contain a complete date.")
        if datetime_component_width < 14 and not injection_time_present:
            raise DataStructureError(
                "Radiopharmaceutical Start DateTime (0018,1078) does not contain a complete time, and "
                "Radiopharmaceutical Start Time (0018,1072) is missing."
            )

    if injection_time_present:
        try:
            injection_clock = parse_time(injection_time_value, "TM")
        except (ValueError, TypeError):
            raise DataStructureError("Radiopharmaceutical Start Time (0018,1072) is invalid.")
        if _time_component_width(injection_time_value) < 6:
            raise DataStructureError("Radiopharmaceutical Start Time (0018,1072) does not contain a complete time.")
    else:
        injection_clock = injection_datetime

    try:
        acquisition_clock = parse_time(ds.AcquisitionTime, "TM")
    except (AttributeError, ValueError, TypeError):
        raise DataStructureError("Acquisition Time (0008,0032) is missing or invalid.")
    if _time_component_width(ds.AcquisitionTime) < 6:
        raise DataStructureError("Acquisition Time (0008,0032) does not contain a complete time.")
    dataset_timezone = _dataset_timezone(ds)
    if injection_datetime is not None and injection_datetime.tzinfo is None and dataset_timezone is not None:
        injection_datetime = injection_datetime.replace(tzinfo=dataset_timezone)
    if injection_clock.tzinfo is None:
        injection_timezone = dataset_timezone
        if injection_timezone is None and injection_datetime is not None:
            injection_timezone = injection_datetime.tzinfo
        if injection_timezone is not None:
            injection_clock = injection_clock.replace(tzinfo=injection_timezone)
    if acquisition_clock.tzinfo is None:
        acquisition_timezone = dataset_timezone
        if acquisition_timezone is None and injection_datetime is not None:
            # Preserve support for legacy datasets containing only one explicit
            # offset on the administration DT.
            acquisition_timezone = injection_datetime.tzinfo
        if acquisition_timezone is not None:
            acquisition_clock = acquisition_clock.replace(tzinfo=acquisition_timezone)

    # A truncated DT defaults omitted fields to zero when parsed. Never treat
    # those defaults as the administration time; use the separately encoded TM.
    if injection_datetime is not None and datetime_component_width < 14:
        injection_datetime = injection_clock.replace(
            year=injection_datetime.year,
            month=injection_datetime.month,
            day=injection_datetime.day,
        )

    # According to the DRO recommendations, long-lived radionuclides require
    # a reliable full administration datetime because uptake can exceed 24 h.
    long_lived_radionuclide = half_life >= 41400
    minimum_reference_offset = 0 if ds.DecayCorrection == "NONE" else -3600

    def validate_reconstructed_times(injection_time, acquisition_time):
        decay_reference_time = get_decay_correction_reference_datetime(
            ds,
            acquisition_time,
            decay_constant,
        )
        reconstructed_offset = (decay_reference_time - injection_time).total_seconds()
        if not minimum_reference_offset <= reconstructed_offset < 2 * half_life:
            raise DataStructureError(
                "Reconstructed administration and acquisition times are inconsistent "
                "with the decay-correction reference datetime."
            )
        return injection_time, acquisition_time

    if not injection_datetime_present and long_lived_radionuclide:
        raise DataStructureError(
            "Radiopharmaceutical Start DateTime (0018,1078) is required "
            "for a long-lived radionuclide because the uptake time may span "
            "more than one day."
        )

    if acquisition_date_present:
        acquisition_date = parse_time(acquisition_date_value, "DA")
        acquisition_time = acquisition_clock.replace(
            year=acquisition_date.year,
            month=acquisition_date.month,
            day=acquisition_date.day,
        )
    else:
        if long_lived_radionuclide:
            raise DataStructureError(
                "Acquisition Date is missing for a long-lived radionuclide. "
                "The uptake time cannot be determined reliably."
            )

        if injection_datetime is not None:
            acquisition_time = acquisition_clock.replace(
                year=injection_datetime.year,
                month=injection_datetime.month,
                day=injection_datetime.day,
            )
        else:
            acquisition_time = acquisition_clock

    # If both complete dates are present, determine whether they are mutually
    # plausible using the actual decay-correction reference datetime.
    if injection_datetime is not None and acquisition_date_present:
        decay_reference_time = get_decay_correction_reference_datetime(
            ds,
            acquisition_time,
            decay_constant,
        )

        datetime_offset = (decay_reference_time - injection_datetime).total_seconds()

        if minimum_reference_offset <= datetime_offset < 2 * half_life:
            return injection_datetime, acquisition_time

        if ds.DecayCorrection == "NONE" and datetime_offset < 0:
            raise DataStructureError(
                "Radiopharmaceutical Start DateTime is after the acquisition datetime for an uncorrected PET frame."
            )

        # An implausible full datetime for a long-lived radionuclide cannot
        # safely be repaired from clock times (DRO_error_4_2).
        if long_lived_radionuclide:
            raise DataStructureError(
                "Radiopharmaceutical Start DateTime is inconsistent with the "
                "decay-correction reference datetime for a long-lived "
                "radionuclide. The administration date may have been anonymized "
                "or shifted and cannot be inferred safely."
            )

    # From here on, the radionuclide is short-lived and either one date is
    # missing or the complete datetime offset was implausible. Ignore the
    # unreliable date offset and reconstruct only the relative day relationship.
    if injection_datetime is not None:
        injection_time = injection_clock.replace(
            year=injection_datetime.year,
            month=injection_datetime.month,
            day=injection_datetime.day,
        )

        acquisition_time = acquisition_clock.replace(
            year=injection_datetime.year,
            month=injection_datetime.month,
            day=injection_datetime.day,
        )

        if injection_time > acquisition_time:
            acquisition_time += timedelta(days=1)

            if (acquisition_time - injection_time).total_seconds() >= 6 * 3600:
                raise DataStructureError(
                    "Injection time is later than acquisition time, but "
                    "assuming a midnight rollover results in a time difference "
                    "of 6 hours or more."
                )

        return validate_reconstructed_times(injection_time, acquisition_time)

    if acquisition_date_present:
        acquisition_date = parse_time(acquisition_date_value, "DA")

        acquisition_time = acquisition_clock.replace(
            year=acquisition_date.year,
            month=acquisition_date.month,
            day=acquisition_date.day,
        )

        injection_time = injection_clock.replace(
            year=acquisition_date.year,
            month=acquisition_date.month,
            day=acquisition_date.day,
        )

        if injection_time > acquisition_time:
            injection_time -= timedelta(days=1)

            if (acquisition_time - injection_time).total_seconds() >= 6 * 3600:
                raise DataStructureError(
                    "Injection time is later than acquisition time while the "
                    "injection date is missing, but assuming injection on the "
                    "previous day results in a time difference of 6 hours or more."
                )

        return validate_reconstructed_times(injection_time, acquisition_time)

    injection_time = injection_clock
    acquisition_time = acquisition_clock

    if injection_time > acquisition_time:
        injection_time -= timedelta(days=1)

        if (acquisition_time - injection_time).total_seconds() >= 6 * 3600:
            raise DataStructureError(
                "Injection time is later than acquisition time while both dates "
                "are missing, but assuming injection on the previous day results "
                "in a time difference of 6 hours or more."
            )

    return validate_reconstructed_times(injection_time, acquisition_time)


def calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time):
    frame_reference_time = (
        _finite_nonnegative(
            ds.FrameReferenceTime,
            "Frame Reference Time (0054,1300)",
        )
        / 1000
    )
    frame_duration = _finite_positive(
        ds.ActualFrameDuration,
        "Actual Frame Duration (0018,1242)",
    )
    decay_during_frame = decay_constant * frame_duration / 1000
    avg_count_rate_time = (1 / decay_constant) * np.log(decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return (acquisition_time - injection_time).total_seconds() + avg_count_rate_time - frame_reference_time


def get_patient_weight_kg(ds):
    """Return patient weight in kg, accepting the IBSI-compatible grams convention."""
    weight = _finite_positive(
        getattr(ds, "PatientWeight", None),
        "Patient Weight (0010,1030)",
    )
    return weight / 1000.0 if weight >= 1000.0 else weight


def get_patient_height_cm(ds):
    """Return patient height in cm."""
    if hasattr(ds, "PatientSize") and ds.PatientSize not in [None, ""]:
        height = _finite_positive(ds.PatientSize, "Patient Size (0010,1020)")
    elif (0x0010, 0x1020) in ds and ds[(0x0010, 0x1020)].value not in [None, ""]:
        height = _finite_positive(ds[(0x0010, 0x1020)].value, "Patient Size (0010,1020)")
    else:
        raise DataStructureError("Patient height tag (0010,1020) is missing.")

    return height * 100.0 if height <= 3 else height


def calculate_bsa_du_bois(height_cm, weight_kg):
    """Calculate Du Bois body surface area in m^2."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute BSA.")
    return 0.007184 * (height_cm**0.725) * (weight_kg**0.425)


def get_patient_sex(ds):
    """Return normalized patient sex: 'M', 'F', or 'O'."""
    sex = getattr(ds, "PatientSex", None)
    if sex is None and (0x0010, 0x0040) in ds:
        sex = ds[(0x0010, 0x0040)].value

    if sex is None:
        raise DataStructureError("Patient sex tag (0010,0040) is missing.")

    sex = str(sex).strip().upper()
    if sex == "":
        raise DataStructureError("Patient sex tag (0010,0040) is empty.")
    if sex not in {"M", "F", "O"}:
        raise DataStructureError(f"Unsupported PatientSex '{sex}'. Expected one of 'M', 'F', or 'O'.")
    return sex


def calculate_lbm_morgan(height_cm, weight_kg, sex):
    """Calculate lean body mass using the Morgan/Sugawara-style formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    male_lbm = 1.10 * weight_kg - 120.0 * ((weight_kg / height_cm) ** 2)
    female_lbm = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed Morgan LBM is non-positive: {lbm}.")
    return lbm


def calculate_lbm_james128(height_cm, weight_kg, sex):
    """Calculate lean body mass using the James/Morgan-128 formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    male_lbm = 1.10 * weight_kg - 128.0 * ((weight_kg / height_cm) ** 2)
    female_lbm = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed James128 LBM is non-positive: {lbm}.")
    return lbm


def calculate_lbm_janmahasatian(height_cm, weight_kg, sex):
    """Calculate lean body mass using the Janmahasatian formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    height_m = height_cm * 1e-2
    bmi = weight_kg / (height_m**2)

    male_lbm = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    female_lbm = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed Janmahasatian LBM is non-positive: {lbm}.")
    return lbm


def calculate_ibw(height_cm, sex):
    """Calculate ideal body weight."""
    if height_cm <= 0:
        raise DataStructureError("Height must be > 0 to compute IBW.")

    male_ibw = 48.0 + 1.06 * (height_cm - 152.0)
    female_ibw = 45.5 + 0.91 * (height_cm - 152.0)

    if sex == "M":
        ibw = male_ibw
    elif sex == "F":
        ibw = female_ibw
    elif sex == "O":
        ibw = 0.5 * (male_ibw + female_ibw)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for IBW calculation.")

    if ibw <= 0:
        raise DataStructureError(f"Computed IBW is non-positive: {ibw}.")
    return ibw


def _gml_suv_type(ds):
    suv_type_elem = ds.get((0x0054, 0x1006), None)
    suv_type = (
        "BW" if suv_type_elem is None or suv_type_elem.value in [None, ""] else str(suv_type_elem.value).strip().upper()
    )
    supported_types = {"BW", "LBM", "LBMJAMES128", "LBMJANMA", "IBW"}
    if suv_type not in supported_types:
        raise DataStructureError(
            f"GML with SUV Type '{suv_type}' is not supported. "
            f"Supported types are BW, LBM, LBMJAMES128, LBMJANMA, and IBW."
        )
    return suv_type


def get_gml_normalization_info(ds):
    """Parse GML SUV normalization metadata and compute the normalization factor."""
    suv_type = _gml_suv_type(ds)

    patient_weight = get_patient_weight_kg(ds)

    if suv_type == "BW":
        return suv_type, patient_weight

    height_cm = get_patient_height_cm(ds)
    sex = get_patient_sex(ds)

    if suv_type == "LBM":
        factor = calculate_lbm_morgan(height_cm, patient_weight, sex)
    elif suv_type == "LBMJAMES128":
        factor = calculate_lbm_james128(height_cm, patient_weight, sex)
    elif suv_type == "LBMJANMA":
        factor = calculate_lbm_janmahasatian(height_cm, patient_weight, sex)
    elif suv_type == "IBW":
        factor = calculate_ibw(height_cm, sex)

    return suv_type, factor


def _enhanced_frame_count(ds):
    try:
        frame_count = int(ds.NumberOfFrames)
    except (AttributeError, TypeError, ValueError):
        raise DataStructureError("Number of Frames (0028,0008) is missing or invalid for Enhanced PET.")
    if frame_count <= 0:
        raise DataStructureError("Number of Frames (0028,0008) must be positive for Enhanced PET.")

    groups = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if groups is None or len(groups) != frame_count:
        raise DataStructureError(
            "Per-Frame Functional Groups Sequence (5200,9230) does not match the number of Enhanced PET frames."
        )
    return frame_count


def _functional_group_sequence(ds, frame_index, sequence_keyword):
    """Resolve a functional-group macro with per-frame precedence."""
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame is not None and frame_index < len(per_frame):
        frame_group = per_frame[frame_index]
        if hasattr(frame_group, sequence_keyword):
            sequence = getattr(frame_group, sequence_keyword)
            if sequence:
                return sequence

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        shared_group = shared[0]
        if hasattr(shared_group, sequence_keyword):
            return getattr(shared_group, sequence_keyword)

    return getattr(ds, sequence_keyword, None)


def _enhanced_radiopharmaceutical_items(ds, frame_index):
    if _is_legacy_converted_enhanced_pet(ds):
        return [_single_radiopharmaceutical_item(ds)]

    sequence = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if sequence is None or len(sequence) == 0:
        raise DataStructureError("Radiopharmaceutical Information Sequence (0054,0016) is missing or empty.")

    usage_sequence = _functional_group_sequence(ds, frame_index, "RadiopharmaceuticalUsageSequence")
    if not usage_sequence:
        if len(sequence) == 1:
            return [sequence[0]]
        raise DataStructureError(
            f"Enhanced PET frame {frame_index + 1} does not identify its radiopharmaceutical agent."
        )

    referenced_agent_numbers = set()
    for usage in usage_sequence:
        try:
            agent_number = int(usage.RadiopharmaceuticalAgentNumber)
        except (AttributeError, TypeError, ValueError):
            raise DataStructureError(
                f"Enhanced PET frame {frame_index + 1} has a missing or invalid Radiopharmaceutical Agent Number "
                "(0018,9729)."
            )
        if agent_number <= 0:
            raise DataStructureError("Radiopharmaceutical Agent Number (0018,9729) must be positive.")
        referenced_agent_numbers.add(agent_number)

    items_by_agent_number = {}
    for rph in sequence:
        try:
            agent_number = int(rph.RadiopharmaceuticalAgentNumber)
        except (AttributeError, TypeError, ValueError):
            raise DataStructureError(
                "Enhanced PET Radiopharmaceutical Information Sequence contains a missing or invalid "
                "Radiopharmaceutical Agent Number (0018,9729)."
            )
        if agent_number in items_by_agent_number:
            raise DataStructureError(f"Enhanced PET has duplicate radiopharmaceutical agent number {agent_number}.")
        items_by_agent_number[agent_number] = rph

    missing_agent_numbers = referenced_agent_numbers - items_by_agent_number.keys()
    if missing_agent_numbers:
        raise DataStructureError(
            f"Enhanced PET frame {frame_index + 1} references radiopharmaceutical agent "
            f"number(s) {sorted(missing_agent_numbers)}, but matching isotope items were not found."
        )
    return [items_by_agent_number[number] for number in sorted(referenced_agent_numbers)]


def _coded_value(code_item):
    """Read a code from any of the three DICOM code-value attributes."""
    for tag in ((0x0008, 0x0100), (0x0008, 0x0119), (0x0008, 0x0120)):
        element = code_item.get(tag)
        if element is not None and element.value not in [None, ""]:
            return str(element.value).strip()
    return None


def _rwvm_unit_code_parts(mapping):
    units_sequence = getattr(mapping, "MeasurementUnitsCodeSequence", None)
    if not units_sequence:
        return None, None
    unit_code = _coded_value(units_sequence[0])
    if unit_code is None:
        return None, None
    compact = re.sub(r"\s+", "", unit_code).replace("\u00b2", "2").casefold()
    return unit_code, re.sub(r"\{[^{}]*\}", "", compact)


def _rwvm_has_recognized_pet_unit(mapping):
    _unit_code, base_unit = _rwvm_unit_code_parts(mapping)
    return base_unit in {"bq/ml", "bqml", "g/ml", "gml", "cm2/ml", "cm2ml"} or (
        base_unit is not None and base_unit.endswith((":bq/ml", ":g/ml", ":cm2/ml"))
    )


def _canonical_suv_type(value):
    if value in [None, ""]:
        return None
    normalized = str(value).strip().upper()
    return normalized if normalized in {"BW", "BSA", "LBM", "LBMJAMES128", "LBMJANMA", "IBW"} else None


def _suv_type_from_unit_code(unit_code):
    annotations = re.findall(r"\{([^{}]+)\}", str(unit_code))
    if not annotations:
        return False, None
    if len(annotations) != 1:
        return True, None
    normalized = re.sub(r"[^A-Z0-9]", "", annotations[0].upper())
    return True, {
        "SUVBW": "BW",
        "SUVBSA": "BSA",
        "SUVLBM": "LBM",
        "SUVLBMJAMES": "LBM",
        "SUVLBMJAMES128": "LBMJAMES128",
        "SUVLBMJANMA": "LBMJANMA",
        "SUVIBW": "IBW",
    }.get(normalized)


def _encoded_suv_type(source, ds, default=None):
    for candidate in (source, ds):
        element = candidate.get((0x0054, 0x1006))
        if element is not None and element.value not in [None, ""]:
            return _canonical_suv_type(element.value)
    return default


def _rwvm_unit_descriptor(mapping, ds):
    unit_code, base_unit = _rwvm_unit_code_parts(mapping)
    if unit_code is None:
        return None

    if base_unit in {"bq/ml", "bqml"} or base_unit.endswith(":bq/ml"):
        return {"kind": "BQML"}

    if base_unit in {"g/ml", "gml"} or base_unit.endswith(":g/ml"):
        has_annotation, suv_type = _suv_type_from_unit_code(unit_code)
        if not has_annotation:
            suv_type = _encoded_suv_type(mapping, ds)
        return {"kind": "SUV", "suv_type": suv_type} if suv_type is not None else None

    if base_unit in {"cm2/ml", "cm2ml"} or base_unit.endswith(":cm2/ml"):
        has_annotation, suv_type = _suv_type_from_unit_code(unit_code)
        if not has_annotation:
            suv_type = _encoded_suv_type(mapping, ds, default="BSA")
        return {"kind": "SUV", "suv_type": suv_type} if suv_type == "BSA" else None
    return None


def _rwvm_mapped_range(mapping, *, require_integer=False):
    first = getattr(
        mapping,
        "RealWorldValueFirstValueMapped",
        getattr(mapping, "DoubleFloatRealWorldValueFirstValueMapped", None),
    )
    last = getattr(
        mapping,
        "RealWorldValueLastValueMapped",
        getattr(mapping, "DoubleFloatRealWorldValueLastValueMapped", None),
    )
    if first in [None, ""] and last in [None, ""]:
        return None
    if first in [None, ""] or last in [None, ""]:
        raise ValueError

    first = float(first)
    last = float(last)
    if not np.isfinite(first) or not np.isfinite(last) or last < first:
        raise ValueError
    if require_integer:
        if not first.is_integer() or not last.is_integer():
            raise ValueError
        return int(first), int(last)
    return first, last


def _rwvm_transform_descriptor(mapping):
    slope = getattr(mapping, "RealWorldValueSlope", None)
    intercept = getattr(mapping, "RealWorldValueIntercept", None)
    lut_data = getattr(mapping, "RealWorldValueLUTData", None)
    if slope not in [None, ""] or intercept not in [None, ""]:
        if lut_data is not None:
            return None
        if slope in [None, ""] or intercept in [None, ""]:
            return None
        try:
            slope = float(slope)
            intercept = float(intercept)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None
        try:
            mapped_range = _rwvm_mapped_range(mapping)
        except (TypeError, ValueError):
            return None
        if mapped_range is None:
            return None
        descriptor = {"method": "linear", "slope": slope, "intercept": intercept}
        descriptor["first"], descriptor["last"] = mapped_range
        return descriptor

    if lut_data is None:
        return None
    try:
        lut = np.asarray(lut_data, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(lut) == 0 or not np.all(np.isfinite(lut)):
        return None

    try:
        mapped_range = _rwvm_mapped_range(mapping, require_integer=True)
    except (TypeError, ValueError):
        return None
    if mapped_range is None:
        return None
    first, last = mapped_range
    if last < first or last - first + 1 != len(lut):
        return None
    return {"method": "lut", "lut": lut, "first": first, "last": last}


def _rwvm_descriptor(mapping, ds):
    unit = _rwvm_unit_descriptor(mapping, ds)
    transform = _rwvm_transform_descriptor(mapping)
    if unit is None or transform is None:
        return None
    return {**unit, **transform}


def _pvt_unit_descriptor(rescale_type, source, ds):
    normalized = str(rescale_type).strip().upper() if rescale_type not in [None, ""] else ""
    if normalized == "BQML":
        return {"kind": "BQML"}
    if normalized == "GML":
        suv_type = _encoded_suv_type(source, ds, default="BW")
        return {"kind": "SUV", "suv_type": suv_type} if suv_type is not None else None
    if normalized == "CM2ML":
        suv_type = _encoded_suv_type(source, ds, default="BSA")
        return {"kind": "SUV", "suv_type": suv_type} if suv_type == "BSA" else None
    if normalized == "CNTS" and _is_legacy_converted_enhanced_pet(ds):
        return {"kind": "CNTS"}
    return None


def _linear_rescale_descriptor(source, ds, unit_value):
    unit = _pvt_unit_descriptor(unit_value, source, ds)
    if unit is None:
        return None
    try:
        slope = float(source.RescaleSlope)
        intercept = float(source.RescaleIntercept)
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return {**unit, "method": "linear", "slope": slope, "intercept": intercept}


def _mapping_value_mask(stored_values, descriptor):
    if "first" not in descriptor:
        return np.ones(stored_values.shape, dtype=bool)
    return (stored_values >= descriptor["first"]) & (stored_values <= descriptor["last"])


def _select_rwvm_candidate(candidates, stored_values):
    ordered_candidates = sorted(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    if stored_values is None:
        return ordered_candidates[0][2]

    grouped_candidates = {}
    for priority, _position, descriptor in ordered_candidates:
        target = (priority, descriptor["kind"], descriptor.get("suv_type"))
        grouped_candidates.setdefault(target, []).append(descriptor)

    for descriptors in grouped_candidates.values():
        complete_descriptor = None
        coverage = np.zeros(stored_values.shape, dtype=bool)
        mapped_values = np.empty(stored_values.shape, dtype=float)
        mappings_agree = True
        for descriptor in descriptors:
            descriptor_mask = _mapping_value_mask(stored_values, descriptor)
            if complete_descriptor is None and np.all(descriptor_mask):
                complete_descriptor = descriptor
            overlap = coverage & descriptor_mask
            if np.any(overlap) and not np.allclose(
                mapped_values[overlap],
                _apply_mapping(stored_values[overlap], descriptor),
                rtol=1e-12,
                atol=1e-12,
            ):
                mappings_agree = False
                break
            newly_mapped = descriptor_mask & ~coverage
            if np.any(newly_mapped):
                mapped_values[newly_mapped] = _apply_mapping(stored_values[newly_mapped], descriptor)
            coverage |= descriptor_mask
        if not np.any(coverage):
            continue
        if not mappings_agree or not np.all(coverage):
            return None
        if complete_descriptor is not None:
            return complete_descriptor
        return {
            "kind": descriptors[0]["kind"],
            "suv_type": descriptors[0].get("suv_type"),
            "method": "composite",
            "mappings": descriptors,
        }
    return None


def _enhanced_frame_mapping(ds, frame_index, stored_values=None):
    rwvm_sequence = _functional_group_sequence(ds, frame_index, "RealWorldValueMappingSequence")
    candidates = []
    if rwvm_sequence:
        for position, mapping in enumerate(rwvm_sequence):
            descriptor = _rwvm_descriptor(mapping, ds)
            if descriptor is None:
                if _rwvm_has_recognized_pet_unit(mapping):
                    raise DataStructureError(
                        f"Enhanced PET frame {frame_index + 1} Real World Value Mapping item {position + 1} "
                        "has recognized quantitative units but an invalid or incomplete transformation."
                    )
                continue
            if descriptor["kind"] == "SUV" and descriptor["suv_type"] == "BW":
                priority = 0
            elif descriptor["kind"] == "SUV":
                priority = 1
            else:
                priority = 2
            candidates.append((priority, position, descriptor))
    if candidates:
        descriptor = _select_rwvm_candidate(candidates, stored_values)
        if descriptor is not None:
            return descriptor
        raise DataStructureError(
            f"Enhanced PET frame {frame_index + 1} has recognized Real World Value Mappings, but they do not "
            "completely and consistently cover its stored pixel values."
        )

    pvt_sequence = _functional_group_sequence(ds, frame_index, "PixelValueTransformationSequence")
    if pvt_sequence:
        for transform in pvt_sequence:
            descriptor = _linear_rescale_descriptor(transform, ds, getattr(transform, "RescaleType", None))
            if descriptor is None and _is_legacy_converted_enhanced_pet(ds):
                descriptor = _linear_rescale_descriptor(transform, ds, getattr(ds, "Units", None))
            if descriptor is not None:
                return descriptor

    if _is_legacy_converted_enhanced_pet(ds):
        # Legacy Converted Enhanced PET may retain the conventional PET attributes.
        unit_value = getattr(ds, "Units", None)
        if _pvt_unit_descriptor(unit_value, ds, ds) is None:
            unit_value = getattr(ds, "RescaleType", None)
        descriptor = _linear_rescale_descriptor(ds, ds, unit_value)
        if descriptor is not None:
            return descriptor

    raise DataStructureError(
        f"Enhanced PET frame {frame_index + 1} has no Real World Value Mapping or Pixel Value Transformation "
        "with sufficient information for conversion to SUV or Bq/ml."
    )


def _apply_mapping(stored_values, descriptor):
    if descriptor["method"] == "linear":
        if not np.all(_mapping_value_mask(stored_values, descriptor)):
            raise DataStructureError("Stored pixel values fall outside the Real World Value Mapping range.")
        return stored_values.astype(float) * descriptor["slope"] + descriptor["intercept"]

    if descriptor["method"] == "composite":
        real_values = np.empty(stored_values.shape, dtype=float)
        unmapped = np.ones(stored_values.shape, dtype=bool)
        for mapping in descriptor["mappings"]:
            selected = unmapped & _mapping_value_mask(stored_values, mapping)
            if np.any(selected):
                real_values[selected] = _apply_mapping(stored_values[selected], mapping)
                unmapped[selected] = False
        if np.any(unmapped):
            raise DataStructureError("Stored pixel values are not covered by the Real World Value Mappings.")
        return real_values

    integer_values = stored_values.astype(np.int64)
    if not np.array_equal(stored_values, integer_values):
        raise DataStructureError("Real World Value LUT Data can only map integer stored pixel values.")
    if np.any(integer_values < descriptor["first"]) or np.any(integer_values > descriptor["last"]):
        raise DataStructureError("Stored pixel values fall outside the Real World Value LUT Data range.")
    return descriptor["lut"][integer_values - descriptor["first"]]


def _suv_to_suvbw_factor(ds, suv_type):
    if suv_type == "BW":
        return 1.0

    patient_weight = get_patient_weight_kg(ds)
    height_cm = get_patient_height_cm(ds)
    if suv_type == "BSA":
        return patient_weight / (calculate_bsa_du_bois(height_cm, patient_weight) * 10.0)

    sex = get_patient_sex(ds)
    if suv_type == "LBM":
        normalization = calculate_lbm_morgan(height_cm, patient_weight, sex)
    elif suv_type == "LBMJAMES128":
        normalization = calculate_lbm_james128(height_cm, patient_weight, sex)
    elif suv_type == "LBMJANMA":
        normalization = calculate_lbm_janmahasatian(height_cm, patient_weight, sex)
    elif suv_type == "IBW":
        normalization = calculate_ibw(height_cm, sex)
    else:
        raise DataStructureError(f"Unsupported Enhanced PET SUV normalization type '{suv_type}'.")
    return patient_weight / normalization


def _parse_complete_datetime(value, name):
    if value in [None, ""]:
        raise DataStructureError(f"{name} is missing or empty.")
    if _time_component_width(value) < 14:
        raise DataStructureError(f"{name} does not contain a complete datetime.")
    try:
        return parse_time(value, "DT")
    except (TypeError, ValueError):
        raise DataStructureError(f"{name} is invalid.")


def _make_datetimes_comparable(ds, first, second):
    dataset_timezone = _dataset_timezone(ds)
    if dataset_timezone is not None:
        if first.tzinfo is None:
            first = first.replace(tzinfo=dataset_timezone)
        if second.tzinfo is None:
            second = second.replace(tzinfo=dataset_timezone)
    elif first.tzinfo is None and second.tzinfo is not None:
        first = first.replace(tzinfo=second.tzinfo)
    elif second.tzinfo is None and first.tzinfo is not None:
        second = second.replace(tzinfo=first.tzinfo)
    return first, second


def _average_activity_time_seconds(duration_ms, decay_constant):
    duration_seconds = duration_ms / 1000.0
    if duration_seconds == 0:
        return 0.0
    decay_during_frame = decay_constant * duration_seconds
    return np.log(decay_during_frame / -np.expm1(-decay_during_frame)) / decay_constant


def _enhanced_decay_reference_datetime(ds, frame_index, decay_constant):
    decay_corrected = str(getattr(ds, "DecayCorrected", "")).strip().upper()
    if decay_corrected not in {"YES", "NO"}:
        raise DataStructureError("Decay Corrected (0018,9758) must be present and equal to YES or NO.")

    if decay_corrected == "YES":
        return _parse_complete_datetime(
            getattr(ds, "DecayCorrectionDateTime", None),
            "Decay Correction DateTime (0018,9701)",
        )

    frame_content_sequence = _functional_group_sequence(ds, frame_index, "FrameContentSequence")
    frame_content = frame_content_sequence[0] if frame_content_sequence else ds
    frame_reference = getattr(frame_content, "FrameReferenceDateTime", None)
    if frame_reference not in [None, ""]:
        return _parse_complete_datetime(frame_reference, "Frame Reference DateTime (0018,9151)")

    acquisition = _parse_complete_datetime(
        getattr(frame_content, "FrameAcquisitionDateTime", None),
        "Frame Acquisition DateTime (0018,9074)",
    )
    duration_ms = _finite_nonnegative(
        getattr(frame_content, "FrameAcquisitionDuration", None),
        "Frame Acquisition Duration (0018,9220)",
    )
    return acquisition + timedelta(seconds=_average_activity_time_seconds(duration_ms, decay_constant))


def _has_enhanced_decay_timing(ds, frame_index):
    decay_corrected = str(getattr(ds, "DecayCorrected", "")).strip().upper()
    rph = _single_radiopharmaceutical_item(ds)
    if getattr(rph, "RadiopharmaceuticalStartDateTime", None) in [None, ""]:
        return False
    if decay_corrected == "YES":
        return getattr(ds, "DecayCorrectionDateTime", None) not in [None, ""]
    if decay_corrected != "NO":
        return False

    frame_content_sequence = _functional_group_sequence(ds, frame_index, "FrameContentSequence")
    frame_content = frame_content_sequence[0] if frame_content_sequence else ds
    if getattr(frame_content, "FrameReferenceDateTime", None) not in [None, ""]:
        return True
    return getattr(frame_content, "FrameAcquisitionDateTime", None) not in [None, ""] and getattr(
        frame_content, "FrameAcquisitionDuration", None
    ) not in [None, ""]


def _uses_legacy_converted_decay_timing(ds, frame_index):
    return _is_legacy_converted_enhanced_pet(ds) and not _has_enhanced_decay_timing(ds, frame_index)


def _enhanced_administration_datetime(ds, reference_datetime, half_life, rph=None):
    if rph is None:
        rph = _single_radiopharmaceutical_item(ds)
    administration = _parse_complete_datetime(
        getattr(rph, "RadiopharmaceuticalStartDateTime", None),
        "Radiopharmaceutical Start DateTime (0018,1078)",
    )
    reference_datetime, administration = _make_datetimes_comparable(ds, reference_datetime, administration)
    decay_corrected = str(getattr(ds, "DecayCorrected", "")).strip().upper()
    minimum_offset_seconds = 0 if decay_corrected == "NO" else -3600
    offset_seconds = (reference_datetime - administration).total_seconds()
    if minimum_offset_seconds <= offset_seconds < 2 * half_life:
        return reference_datetime, administration

    if half_life >= 41400:
        raise DataStructureError(
            "Radiopharmaceutical Start DateTime is inconsistent with the decay-correction reference datetime "
            "for a long-lived radionuclide."
        )

    administration = administration.replace(
        year=reference_datetime.year,
        month=reference_datetime.month,
        day=reference_datetime.day,
    )
    if (reference_datetime - administration).total_seconds() < -3600:
        administration -= timedelta(days=1)

    reconstructed_offset_seconds = (reference_datetime - administration).total_seconds()
    if minimum_offset_seconds <= reconstructed_offset_seconds < 2 * half_life:
        return reference_datetime, administration
    raise DataStructureError(
        "Radiopharmaceutical Start DateTime remains inconsistent with the decay-correction reference datetime "
        "after date reconstruction."
    )


def _radiopharmaceutical_name(rph):
    value = getattr(rph, "Radiopharmaceutical", None)
    if value not in [None, ""]:
        return str(value)
    code_sequence = getattr(rph, "RadiopharmaceuticalCodeSequence", None)
    if code_sequence:
        meaning = getattr(code_sequence[0], "CodeMeaning", None)
        if meaning not in [None, ""]:
            return str(meaning)
    return None


def _is_ibsi_suv_dro(ds):
    manufacturer = str(getattr(ds, "Manufacturer", "")).strip().upper()
    institution = str(getattr(ds, "InstitutionName", "")).strip().upper()
    study_description = str(getattr(ds, "StudyDescription", "")).strip().upper()
    return (
        manufacturer == "SYNTHETIC"
        and "IMAGE BIOMARKER STANDARDISATION INITIATIVE" in institution
        and study_description.startswith("PET SUV VERIFICATION DRO_")
    )


def _enhanced_bqml_context(ds, frame_index, require_half_life=True, warn_on_unit_conversion=False):
    radiopharmaceuticals = _enhanced_radiopharmaceutical_items(ds, frame_index)
    if len(radiopharmaceuticals) != 1:
        raise DataStructureError(
            f"Enhanced PET frame {frame_index + 1} references multiple radiopharmaceutical agents, so its Bq/ml "
            "values cannot be converted to a uniquely defined SUV. An encoded SUV Real World Value Mapping is "
            "required for this frame."
        )

    administrations = []
    for rph in radiopharmaceuticals:
        injected_dose = _finite_positive(
            getattr(rph, "RadionuclideTotalDose", None),
            "Radionuclide Total Dose (0018,1074)",
        )
        if not _is_legacy_converted_enhanced_pet(ds):
            if _is_ibsi_suv_dro(ds):
                if warn_on_unit_conversion:
                    warnings.warn(
                        "The IBSI Enhanced PET DRO encodes Radionuclide Total Dose in Bq despite the DICOM MBq "
                        "definition; treating it as Bq.",
                        DataStructureWarning,
                    )
            else:
                injected_dose *= 1000000
        else:
            tracer_name = _radiopharmaceutical_name(rph)
            if tracer_name is not None and is_fdg(tracer_name) and injected_dose < 10000:
                injected_dose *= 1000000
                if warn_on_unit_conversion:
                    warnings.warn(
                        f"Injected dose is {injected_dose} Bq, it is too low for FDG, assumed to be in MBq",
                        DataStructureWarning,
                    )
        half_life = (
            _finite_positive(
                getattr(rph, "RadionuclideHalfLife", None),
                "Radionuclide Half Life (0018,1075)",
            )
            if require_half_life
            else None
        )
        administrations.append({"rph": rph, "injected_dose": injected_dose, "half_life": half_life})
    return {"patient_weight": get_patient_weight_kg(ds), "administrations": administrations}


def _enhanced_frame_administration_context(ds, frame_index, half_life, rph):
    if half_life is None:
        return None
    if _uses_legacy_converted_decay_timing(ds, frame_index):
        if str(getattr(ds, "DecayCorrection", "")).strip().upper() == "ADMIN":
            return None
        decay_constant = np.log(2) / half_life
        administration_datetime, _acquisition_datetime = resolve_injection_and_acquisition_times(
            ds,
            half_life,
            decay_constant,
        )
        return administration_datetime

    decay_constant = np.log(2) / half_life
    reference_datetime = _enhanced_decay_reference_datetime(ds, frame_index, decay_constant)
    _reference_datetime, administration_datetime = _enhanced_administration_datetime(
        ds,
        reference_datetime,
        half_life,
        rph=rph,
    )
    return administration_datetime


def _coded_sequence_identity(item, sequence_keyword):
    sequence = getattr(item, sequence_keyword, None)
    if not sequence:
        return "unspecified"
    code_item = sequence[0]
    code_value = _coded_value(code_item)
    scheme = str(getattr(code_item, "CodingSchemeDesignator", "")).strip().upper()
    return f"{scheme}:{code_value}" if code_value is not None else "unspecified"


def _enhanced_radiopharmaceutical_identity(ds, rph):
    base_identity = (
        _coded_sequence_identity(rph, "RadionuclideCodeSequence"),
        _coded_sequence_identity(rph, "RadiopharmaceuticalCodeSequence"),
    )
    sequence = getattr(ds, "RadiopharmaceuticalInformationSequence", None) or []
    matching = [
        (str(getattr(candidate, "RadiopharmaceuticalStartDateTime", "")).strip(), position, candidate)
        for position, candidate in enumerate(sequence)
        if (
            _coded_sequence_identity(candidate, "RadionuclideCodeSequence"),
            _coded_sequence_identity(candidate, "RadiopharmaceuticalCodeSequence"),
        )
        == base_identity
    ]
    matching.sort(key=lambda entry: (entry[0], entry[1]))
    occurrence = next(
        (index for index, (_start, _position, candidate) in enumerate(matching, 1) if candidate is rph), 1
    )
    identity = f"radionuclide {base_identity[0]}, radiopharmaceutical {base_identity[1]}"
    return f"{identity}, administration {occurrence}" if len(matching) > 1 else identity


def _enhanced_radiopharmaceutical_context_key(ds, rph, name):
    if _is_legacy_converted_enhanced_pet(ds):
        return name
    return f"{name} [{_enhanced_radiopharmaceutical_identity(ds, rph)}]"


def _store_normalization_context_value(context, name, value):
    if name in context:
        reference = context[name]
        if isinstance(value, (int, float, np.number)) and isinstance(reference, (int, float, np.number)):
            values_agree = np.isclose(value, reference, rtol=1e-12, atol=0.0)
        else:
            values_agree = value == reference
        if not values_agree:
            raise DataStructureError(f"Enhanced PET frames have inconsistent {name} values.")
        return
    context[name] = value


def _legacy_converted_bqml_to_suvbw_factor(ds, context):
    patient_weight = context["patient_weight"]
    administration = context["administrations"][0]
    injected_dose = administration["injected_dose"]
    half_life = administration["half_life"]
    decay_correction = str(getattr(ds, "DecayCorrection", "")).strip().upper()
    if decay_correction == "ADMIN":
        return patient_weight * 1000.0 / injected_dose
    if decay_correction not in {"START", "NONE"}:
        raise DataStructureError(
            "Legacy Converted Enhanced PET requires either Decay Corrected (0018,9758) metadata or a supported "
            "conventional Decay Correction (0054,1102) value."
        )

    decay_constant = np.log(2) / half_life
    injection_time, acquisition_time = resolve_injection_and_acquisition_times(ds, half_life, decay_constant)
    if decay_correction == "START":
        elapsed_time = calc_start_elapsed_time(ds, decay_constant, acquisition_time, injection_time)
    else:
        frame_duration = _finite_positive(
            getattr(ds, "ActualFrameDuration", None),
            "Actual Frame Duration (0018,1242)",
        )
        elapsed_time = (acquisition_time - injection_time).total_seconds() + _average_activity_time_seconds(
            frame_duration,
            decay_constant,
        )
    decay_corrected_dose = injected_dose * np.exp(-decay_constant * elapsed_time)
    return patient_weight * 1000.0 / decay_corrected_dose


def _enhanced_bqml_to_suvbw_factor(ds, frame_index, context):
    if _uses_legacy_converted_decay_timing(ds, frame_index):
        return _legacy_converted_bqml_to_suvbw_factor(ds, context)

    administration = context["administrations"][0]
    half_life = administration["half_life"]
    decay_constant = np.log(2) / half_life
    reference_datetime = _enhanced_decay_reference_datetime(ds, frame_index, decay_constant)
    reference_datetime, administration_datetime = _enhanced_administration_datetime(
        ds,
        reference_datetime,
        half_life,
        rph=administration["rph"],
    )
    elapsed_time = (reference_datetime - administration_datetime).total_seconds()
    decay_corrected_dose = administration["injected_dose"] * np.exp(-decay_constant * elapsed_time)
    return context["patient_weight"] * 1000.0 / decay_corrected_dose


def _enhanced_cnts_scale(ds):
    manufacturer = str(getattr(ds, "Manufacturer", "")).upper()
    if "PHILIPS" not in manufacturer:
        raise DataStructureError("Conventional CNTS fallback is only supported for Philips PET images.")
    if (0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0:
        return "BQML", _finite_positive(ds[(0x7053, 0x1009)].value, "Philips scale factor (7053,1009)")
    if (
        str(getattr(ds, "DecayCorrection", "")).strip().upper() != "NONE"
        and (0x7053, 0x1000) in ds
        and ds[(0x7053, 0x1000)].value != 0
    ):
        return "SUV", _finite_positive(ds[(0x7053, 0x1000)].value, "Philips scale factor (7053,1000)")
    raise DataStructureError("Philips-specific scaling factors are missing for Legacy Converted Enhanced PET CNTS.")


def _descriptor_uses_bqml(ds, descriptor):
    if descriptor["kind"] == "BQML":
        return True
    return descriptor["kind"] == "CNTS" and _enhanced_cnts_scale(ds)[0] == "BQML"


def _bqml_frames_require_half_life(ds, frame_indices):
    decay_correction = str(getattr(ds, "DecayCorrection", "")).strip().upper()
    return any(
        not _uses_legacy_converted_decay_timing(ds, frame_index) or decay_correction != "ADMIN"
        for frame_index in frame_indices
    )


def _enhanced_descriptor_to_suvbw_factor(ds, frame_index, descriptor, bqml_context):
    if descriptor["kind"] == "SUV":
        return _suv_to_suvbw_factor(ds, descriptor["suv_type"])
    if descriptor["kind"] == "BQML":
        return _enhanced_bqml_to_suvbw_factor(ds, frame_index, bqml_context)

    scale_kind, scale = _enhanced_cnts_scale(ds)
    if scale_kind == "SUV":
        return scale
    return scale * _enhanced_bqml_to_suvbw_factor(ds, frame_index, bqml_context)


def _validate_enhanced_pet(ds):
    frame_count = _enhanced_frame_count(ds)
    # Pixel ranges determine which of several RWVM items is applicable. The
    # metadata-only dataset used here cannot safely validate normalization or
    # dose requirements for one candidate before the stored values are read.
    for frame_index in range(frame_count):
        _enhanced_frame_mapping(ds, frame_index)


def _enhanced_suv_array(ds, return_normalization_context=False):
    frame_count = _enhanced_frame_count(ds)
    stored_frames = np.asarray(ds.pixel_array)
    if stored_frames.ndim == 2:
        stored_frames = stored_frames[np.newaxis, ...]
    if stored_frames.ndim != 3 or stored_frames.shape[0] != frame_count:
        raise DataStructureError("Enhanced PET pixel data dimensions do not match Number of Frames (0028,0008).")

    descriptors = [
        _enhanced_frame_mapping(ds, frame_index, stored_frames[frame_index]) for frame_index in range(frame_count)
    ]
    bqml_frame_indices = [
        frame_index for frame_index, descriptor in enumerate(descriptors) if _descriptor_uses_bqml(ds, descriptor)
    ]
    bqml_contexts = {
        frame_index: _enhanced_bqml_context(
            ds,
            frame_index,
            require_half_life=_bqml_frames_require_half_life(ds, [frame_index]),
            warn_on_unit_conversion=frame_index == bqml_frame_indices[0],
        )
        for frame_index in bqml_frame_indices
    }
    normalization_context = {}
    for frame_index, bqml_context in bqml_contexts.items():
        _store_normalization_context_value(
            normalization_context,
            "Patient Weight (0010,1030)",
            bqml_context["patient_weight"],
        )
        for administration in bqml_context["administrations"]:
            rph = administration["rph"]
            injected_dose = administration["injected_dose"]
            half_life = administration["half_life"]
            dose_name = _enhanced_radiopharmaceutical_context_key(
                ds,
                rph,
                "Radionuclide Total Dose (0018,1074)",
            )
            _store_normalization_context_value(normalization_context, dose_name, injected_dose)
            if half_life is not None:
                half_life_name = _enhanced_radiopharmaceutical_context_key(
                    ds,
                    rph,
                    "Radionuclide Half Life (0018,1075)",
                )
                _store_normalization_context_value(normalization_context, half_life_name, half_life)

            administration_datetime = _enhanced_frame_administration_context(ds, frame_index, half_life, rph)
            if administration_datetime is not None:
                administration_name = _enhanced_radiopharmaceutical_context_key(
                    ds,
                    rph,
                    "Radiopharmaceutical Start DateTime (0018,1078)",
                )
                _store_normalization_context_value(
                    normalization_context,
                    administration_name,
                    administration_datetime,
                )

    non_bw_suv_types = {
        descriptor["suv_type"]
        for descriptor in descriptors
        if descriptor["kind"] == "SUV" and descriptor["suv_type"] != "BW"
    }
    if non_bw_suv_types:
        normalization_context["Patient Weight (0010,1030)"] = get_patient_weight_kg(ds)
        normalization_context["Patient Size (0010,1020)"] = get_patient_height_cm(ds)
        if non_bw_suv_types != {"BSA"}:
            normalization_context["Patient Sex (0010,0040)"] = get_patient_sex(ds)

    suv_frames = np.empty(stored_frames.shape, dtype=float)
    for frame_index, (stored_frame, descriptor) in enumerate(zip(stored_frames, descriptors)):
        physical_values = _apply_mapping(stored_frame, descriptor)
        factor = _enhanced_descriptor_to_suvbw_factor(
            ds,
            frame_index,
            descriptor,
            bqml_contexts.get(frame_index),
        )
        suv_frames[frame_index] = physical_values * factor
    if return_normalization_context:
        return suv_frames, normalization_context
    return suv_frames


def _validate_enhanced_normalization_contexts(contexts):
    reference_values = {}
    reference_bqml_fields = None
    bqml_field_prefixes = (
        "Radionuclide Total Dose (0018,1074)",
        "Radionuclide Half Life (0018,1075)",
        "Radiopharmaceutical Start DateTime (0018,1078)",
    )
    for instance_index, context in enumerate(contexts):
        bqml_fields = {name for name in context if any(name.startswith(prefix) for prefix in bqml_field_prefixes)}
        if bqml_fields:
            if reference_bqml_fields is None:
                reference_bqml_fields = bqml_fields
            elif bqml_fields != reference_bqml_fields:
                raise DataStructureError(
                    f"Enhanced PET instance {instance_index + 1} uses a different radiopharmaceutical "
                    "administration context for Bq/ml SUV normalization."
                )

        for name, value in context.items():
            reference_value = reference_values.get(name)
            if isinstance(value, (int, float, np.number)) and isinstance(reference_value, (int, float, np.number)):
                values_agree = np.isclose(value, reference_value, rtol=1e-12, atol=0.0)
            else:
                values_agree = value == reference_value
            if name in reference_values and not values_agree:
                raise DataStructureError(
                    f"Enhanced PET instances have inconsistent {name} values; instance {instance_index + 1} "
                    "would use a different SUV normalization factor."
                )
            reference_values.setdefault(name, value)


def validate_pet_dicom_tags(dicom_files):
    enhanced_flags = [is_enhanced_pet(dcm_file["ds"]) for dcm_file in dicom_files]
    if any(enhanced_flags):
        if not all(enhanced_flags):
            raise DataStructureError("A PET series cannot mix Enhanced and conventional PET instances.")
        for dcm_file in dicom_files:
            _validate_enhanced_pet(dcm_file["ds"])
        return

    for dcm_file in dicom_files:
        ds = dcm_file["ds"]
        image_id = dcm_file["file_path"]

        patient_weight = get_patient_weight_kg(ds)
        if patient_weight < 1:
            warning_msg = f"For patient's {image_id} image, patient's weight tag (0010,1030) contains weight < 1kg. Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)
        if "DECY" not in ds[(0x0028, 0x0051)].value or "ATTN" not in ds[(0x0028, 0x0051)].value:
            warning_msg = f"For patient's {image_id} image, in DICOM tag (0028, 0051) either no 'DECY' (decay correction) or 'ATTN' (attenuation correction). Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)
        if ds.Units == "BQML":
            rph = _single_radiopharmaceutical_item(ds)
            _finite_positive(
                getattr(rph, "RadionuclideTotalDose", None),
                "Radionuclide Total Dose (0018,1074)",
            )

            if ds.DecayCorrection != "ADMIN":
                half_life = get_radionuclide_half_life(ds)
                decay_constant = np.log(2) / half_life
                injection_time, acquisition_time = resolve_injection_and_acquisition_times(
                    ds,
                    half_life,
                    decay_constant,
                )

            if ds.DecayCorrection == "START":
                elapsed_time = calc_start_elapsed_time(
                    ds,
                    decay_constant,
                    acquisition_time,
                    injection_time,
                )
                manufacturer = ds.Manufacturer.upper()
                if not any(vendor in manufacturer for vendor in ("PHILIPS", "SIEMENS", "CPS", "CTI", "GE")):
                    warning_msg = f"For patient's {image_id} image, an unknown PET scaner manufacturer is present {ds.Manufacturer}. Siemens/Philips strategy is applied."
                    warnings.warn(warning_msg, DataStructureWarning)

            elif ds.DecayCorrection == "ADMIN":
                elapsed_time = 0
            elif ds.DecayCorrection == "NONE":
                elapsed_time = None
            else:
                warning_msg = f"For patient's {image_id} image, An unsupported Decay Correction {ds.DecayCorrection} is present. Only supported are 'NONE', 'START' and 'ADMIN'. Patient is excluded from the analysis."
                raise DataStructureError(warning_msg)
            if elapsed_time is not None and elapsed_time < 0:
                error_msg = f"For patient's {image_id} image, patient is excluded from the analysis due to the negative time difference in the decay factor."
                raise DataStructureError(error_msg)
            elif (
                elapsed_time is not None
                and elapsed_time > 0
                and abs(elapsed_time) < 1800
                and ds.DecayCorrection != "ADMIN"
            ):
                warning_msg = f"Only {abs(elapsed_time) / 60} minutes after the injection."
                warnings.warn(warning_msg, DataStructureWarning)
        elif ds.Units == "CNTS" and "PHILIPS" in ds.Manufacturer.upper():
            has_bqml_scale = (0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0
            has_direct_scale = (
                ds.DecayCorrection != "NONE" and (0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0
            )
            if not has_bqml_scale and not has_direct_scale:
                error_msg = f"For patient's {image_id} image, patient is excluded, Philips scale factors not present (PET units CNTS)"
                raise DataStructureError(error_msg)
            scale_tag = (0x7053, 0x1009) if has_bqml_scale else (0x7053, 0x1000)
            _finite_positive(ds[scale_tag].value, f"Philips scale factor {scale_tag}")
        elif ds.Units == "GML":
            try:
                _suv_type, _factor = get_gml_normalization_info(ds)
            except Exception as e:
                error_msg = f"For patient's {image_id} image, patient is excluded, GML normalization is invalid: {e}"
                raise DataStructureError(error_msg)

        elif ds.Units == "CM2ML":
            suv_type = ds.get((0x0054, 0x1006), None)
            if suv_type is not None and suv_type.value != "BSA":
                error_msg = f"For patient's {image_id} image, patient is excluded, SUV Type is not BSA (CM2ML units)"
                raise DataStructureError(error_msg)

            try:
                patient_weight = get_patient_weight_kg(ds)
                height_cm = get_patient_height_cm(ds)
                _ = calculate_bsa_du_bois(height_cm, patient_weight)
            except Exception as e:
                error_msg = (
                    f"For patient's {image_id} image, CM2ML requires valid patient "
                    f"height and weight for Du Bois BSA calculation: {e}"
                )
                raise DataStructureError(error_msg)

        else:
            error_msg = f"For patient's {image_id} image, patient is excluded, only supported PET Units are BQML for Philips, Siemens and GE or CNTS for Philips"
            raise DataStructureError(error_msg)


def apply_suv_correction(dicom_files, suv_image):
    enhanced_flags = [is_enhanced_pet(dcm_file["ds"]) for dcm_file in dicom_files]
    if any(enhanced_flags):
        if not all(enhanced_flags):
            raise DataStructureError("A PET series cannot mix Enhanced and conventional PET instances.")
        try:
            instance_results = [
                _enhanced_suv_array(
                    pydicom.dcmread(dcm_file["file_path"]),
                    return_normalization_context=True,
                )
                for dcm_file in dicom_files
            ]
            _validate_enhanced_normalization_contexts([context for _array, context in instance_results])
            intensity_array = np.concatenate([array for array, _context in instance_results], axis=0)
        except ValueError as exc:
            raise DataStructureError("Enhanced PET instances have incompatible pixel dimensions.") from exc

        size = suv_image.GetSize()
        expected_shape = (size[2], size[1], size[0])
        if intensity_array.shape != expected_shape:
            raise DataStructureError(
                f"Enhanced PET SUV array shape {intensity_array.shape} does not match image shape {expected_shape}."
            )
        intensity_image = sitk.GetImageFromArray(intensity_array)
        intensity_image.CopyInformation(suv_image)
        return intensity_image

    def process_single_slice(dicom_file_path):
        ds = pydicom.dcmread(dicom_file_path)

        def get_tracer_name(rph_item):
            value = getattr(rph_item, "Radiopharmaceutical", None)
            if value is None and (0x0018, 0x0031) in rph_item:
                value = rph_item[(0x0018, 0x0031)].value
            return str(value) if value is not None else None

        def process_gml(pixel_array_units, ds):
            suv_type, factor = get_gml_normalization_info(ds)
            patient_weight = get_patient_weight_kg(ds)

            if suv_type == "BW":
                return pixel_array_units

            return pixel_array_units * (patient_weight / factor)

        def process_cm2ml(pixel_array_units, ds):
            suv_type = ds.get((0x0054, 0x1006), None)
            if suv_type is not None and suv_type.value != "BSA":
                raise DataStructureError(f"CM2ML with {suv_type.value} SUV normalization is not supported!")

            patient_weight = get_patient_weight_kg(ds)
            height_cm = get_patient_height_cm(ds)
            bsa_m2 = calculate_bsa_du_bois(height_cm, patient_weight)

            return pixel_array_units * (patient_weight / (bsa_m2 * 10.0))

        def process_bqml(activity_concentration, ds):
            rph = _single_radiopharmaceutical_item(ds)
            patient_weight = get_patient_weight_kg(ds)
            injected_dose = _finite_positive(
                getattr(rph, "RadionuclideTotalDose", None),
                "Radionuclide Total Dose (0018,1074)",
            )

            tracer_name = get_tracer_name(rph)
            if tracer_name is not None and is_fdg(tracer_name) and injected_dose < 10000:
                injected_dose *= 1000000
                warnings.warn(
                    f"Injected dose is {injected_dose} Bq, it is too low for FDG, assumed to be in MBq",
                    DataStructureWarning,
                )

            decay_correction = ds.DecayCorrection

            if decay_correction == "ADMIN":
                return activity_concentration / (injected_dose / (patient_weight * 1000))

            half_life = get_radionuclide_half_life(ds)
            decay_constant = np.log(2) / half_life
            injection_time, acquisition_time = resolve_injection_and_acquisition_times(
                ds,
                half_life,
                decay_constant,
            )

            if decay_correction == "START":
                elapsed_time = calc_start_elapsed_time(
                    ds,
                    decay_constant,
                    acquisition_time,
                    injection_time,
                )
                decay_factor = np.exp(-(np.log(2) * elapsed_time) / half_life)
                decay_corrected_dose = injected_dose * decay_factor
                return activity_concentration / (decay_corrected_dose / (patient_weight * 1000))

            if decay_correction == "NONE":
                frame_duration = _finite_positive(
                    ds.ActualFrameDuration,
                    "Actual Frame Duration (0018,1242)",
                )
                decay_during_frame = decay_constant * frame_duration / 1000.0
                avg_count_rate_time = (1 / decay_constant) * np.log(
                    decay_during_frame / (1 - np.exp(-decay_during_frame))
                )
                decay_corrected_activity_concentration = activity_concentration * np.exp(
                    decay_constant * ((acquisition_time - injection_time).total_seconds() + avg_count_rate_time)
                )

                return decay_corrected_activity_concentration / (injected_dose / (patient_weight * 1000))

            raise DataStructureError(f"Decay correction {decay_correction} is not supported!")

        def process_cnts(pixel_array_units, ds):
            manufacturer = ds.Manufacturer.upper()
            if "PHILIPS" not in manufacturer:
                raise DataStructureError(f"Vendor {ds.Manufacturer} is not supported with CNTS units!")

            if (0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0:
                scale = _finite_positive(ds[(0x7053, 0x1009)].value, "Philips scale factor (7053,1009)")
                activity_concentration_bqml = pixel_array_units * scale
                return process_bqml(activity_concentration_bqml, ds)

            if ds.DecayCorrection != "NONE" and (0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0:
                scale = _finite_positive(ds[(0x7053, 0x1000)].value, "Philips scale factor (7053,1000)")
                return pixel_array_units * scale

            raise DataStructureError("Philips-specific scaling factors not present!")

        units = ds.Units
        try:
            rescale_slope = float(ds.RescaleSlope)
        except (AttributeError, TypeError, ValueError):
            raise DataStructureError("Rescale Slope (0028,1053) is missing or invalid.")
        if not np.isfinite(rescale_slope):
            raise DataStructureError("Rescale Slope (0028,1053) must be finite.")
        if rescale_slope <= 0:
            warnings.warn(
                "Rescale Slope (0028,1053) is non-positive; applying it as encoded.",
                DataStructureWarning,
            )
        try:
            rescale_intercept = float(ds.RescaleIntercept)
        except (TypeError, ValueError):
            raise DataStructureError("Rescale Intercept (0028,1052) is missing or invalid.")
        if not np.isfinite(rescale_intercept):
            raise DataStructureError("Rescale Intercept (0028,1052) must be finite.")
        pixel_array_units = (ds.pixel_array * rescale_slope) + rescale_intercept

        if units == "GML":
            suv = process_gml(pixel_array_units, ds)
        elif units == "CM2ML":
            suv = process_cm2ml(pixel_array_units, ds)
        elif units == "BQML":
            suv = process_bqml(pixel_array_units, ds)
        elif units == "CNTS":
            suv = process_cnts(pixel_array_units, ds)
        else:
            raise DataStructureError(f"Units {units} are not supported!")

        return suv.T

    intensity_array = np.zeros(suv_image.GetSize())

    dicom_files = sorted(dicom_files, key=lambda f: float(f["ds"].ImagePositionPatient[2]))
    for z_slice_id, dicom_file in enumerate(dicom_files):
        intensity_array[:, :, z_slice_id] = process_single_slice(dicom_file["file_path"])

    intensity_image = sitk.GetImageFromArray(intensity_array.T)
    intensity_image.SetOrigin(suv_image.GetOrigin())
    intensity_image.SetSpacing(np.array(suv_image.GetSpacing()))
    intensity_image.SetDirection(np.array(suv_image.GetDirection()))
    return intensity_image