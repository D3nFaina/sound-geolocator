"""Rough ICAO type-designator classification: propeller, jet or helicopter.

Propeller aircraft (turboprops, pistons) and helicopters have strong, stable
blade-passing tones and fly low, so they are usually the clearest Doppler
sources. Filtering candidate flights by class removes many wrong matches.

This list covers common types; anything unknown is classed "unknown" and is
kept unless you filter it out explicitly. Add types with :data:`EXTRA`.
"""

from __future__ import annotations

PROP = {
    # regional turboprops
    "AT43", "AT44", "AT45", "AT46", "AT72", "AT73", "AT75", "AT76", "DH8A", "DH8B", "DH8C", "DH8D",
    "SF34", "SB20", "F50", "F27", "JS31", "JS32", "JS41", "E110", "E120", "D228", "D328", "SW4", "SW3",
    "L410", "AN24", "AN26", "AN28", "AN32", "B190", "C212", "CN35", "C295", "Y12", "MA60", "SH36", "DHC6",
    "DHC7", "DHC2", "DHC3", "AN2", "AN12", "C130", "C30J", "P3", "E314", "A400", "C160",
    # business / utility turboprops
    "B350", "BE20", "BE30", "BE9L", "BE9T", "BE99", "BE10", "C208", "C425", "C441", "C406", "PC12", "PC6T",
    "PC7", "PC9", "PC21", "TBM7", "TBM8", "TBM9", "P180", "PAY1", "PAY2", "PAY3", "PAY4", "MU2", "AC90",
    "AC95", "EPIC", "M600", "KODI", "SC7", "BN2T", "AT8T", "AT3T", "AT5T", "AT6T", "AT8T",
    # pistons
    "C150", "C152", "C162", "C170", "C172", "C175", "C177", "C180", "C182", "C185", "C195", "C206", "C207",
    "C210", "C310", "C340", "C337", "C402", "C414", "C421", "P28A", "P28B", "P28R", "P28T", "PA18",
    "PA20", "PA22", "PA24", "PA27", "PA30", "PA31", "PA32", "PA34", "PA38", "PA44", "PA46", "P32R", "P46T",
    "SR20", "SR22", "S22T", "DA20", "DA40", "DA42", "DA62", "BE33", "BE35", "BE36", "BE55", "BE58",
    "BE76", "BE23", "BE24", "M20P", "M20T", "M20J", "RV4", "RV6", "RV7", "RV8", "RV9", "RV10", "RV12",
    "TB10", "TB20", "TB21", "DR40", "DR10", "J3", "CH7A", "CUB", "AA5", "AA1", "G115", "TOBA", "SIRA",
    "ULAC", "GLID", "C42", "P92", "WT9", "ATL", "VIRU", "BN2P", "AEST", "GA8", "DHC1",
}

HELI = {
    "EC20", "EC25", "EC30", "EC35", "EC45", "EC55", "EC75", "H160", "H175", "AS32", "AS50", "AS55",
    "AS65", "AS3B", "A109", "A119", "A139", "A149", "A169", "A189", "B06", "B06T", "B105", "B212",
    "B222", "B230", "B407", "B412", "B427", "B429", "B430", "B505", "R22", "R44", "R66", "S76", "S92",
    "S61", "H60", "UH1", "MI8", "MI17", "MI2", "NH90", "EH10", "BK17", "H269", "H500", "MD52", "MD60",
    "EXPL", "GAZL", "ALO3", "LYNX", "PUMA", "KA32", "CH47", "H47", "AW09",
}

JET_PREFIXES = ("A3", "A2", "B7", "B3", "BCS", "E1", "E5", "E7", "E9", "E2", "E3", "CRJ", "C25", "C5",
                "C6", "C7", "CL3", "CL6", "GLF", "GLEX", "GL5", "GL7", "F2", "F9", "FA", "LJ", "H25", "PC24",
                "HDJT", "SF50", "EA50", "C510", "PRM1", "MD8", "MD9", "MD1", "DC9", "DC10", "B46", "RJ",
                "F70", "F100", "SU95", "A1", "IL", "T1", "T2", "AN1", "AN7", "C17", "K35", "E135", "E145",
                "E170", "E190", "E195", "E290", "E295", "A19", "A20", "A21", "A22", "A31", "A32", "A33",
                "A34", "A35", "A38", "B37", "B38", "B39", "B73", "B74", "B75", "B76", "B77", "B78", "C68",
                "C56", "C55", "E50", "E55", "J328", "BE40", "G150", "G280", "ASTR", "WW24", "GALX")

EXTRA: dict[str, str] = {}


def classify(typecode: str | None) -> str:
    """Return "prop", "jet", "heli" or "unknown" for an ICAO type designator."""
    if not typecode:
        return "unknown"
    t = typecode.strip().upper()
    if t in EXTRA:
        return EXTRA[t]
    if t in HELI:
        return "heli"
    if t in PROP:
        return "prop"
    if t.startswith(JET_PREFIXES):
        return "jet"
    return "unknown"


def filter_tracks(tracks, keep=("prop", "heli", "unknown")):
    """Keep only flights whose class is in ``keep``."""
    keep = set(keep)
    return [tr for tr in tracks if classify(getattr(tr, "typecode", "")) in keep]
