"""
Alle Funktionen/Module die das Laden, speichern und verarbeiten der Daten betreffen.
"""
from abc import ABC


class Attrs(ABC):
    """Einfacher Container für alle Attributnamen.
    Ergebnis der :ref:`Attributanalyse`, kodiert als statische Klassenattribute.
    """

    #: Qualitativ-nominale Attribute
    quali_nominal = [
        'AMLSTAT', 'CD34SRC', 'WHO', 'CISTAT', 'CEBPASTAT', 'IDH1T', 'IDH2T', 'FAB']

    #: Qualitativ-ordinale Attribute
    quali_ordinal = [
        'ECOG', 'ELNRisk', 'CGELN', 'CGSTUD']

    #: Qualitative Binärattribute
    quali_bin = [
        'FEV', 'SEX', 'RFSSTAT', 'EFSSTAT',
        'OSSTAT', 'CR1', 'ED30', 'ED60', 'CGCX', 'CGNK']

    #: Quantitativ-diskrete Attribute
    quant_discrete = [
        'AGE', 'PBB', 'BMB', 'D16BMB']

    #: Quantitativ-kontinuierliche Attribute
    quant_continuous = [
        'RFSTM', 'OSTM', 'EFSTM', 'CD34', 'WBC', 'PLT', 'LDH', 'HB', 'FIB', 'FLT3R']

    #: Qualitativ-nominale Mutationsindikatoren
    quali_nominal_mutation_indicators = [
        'CEBPA', 'CEBPADM', 'FLT3I', 'FLT3T', 'IDH1', 'IDH2', 'JAK2', 'NPM1', 'EXAML']

    #: Quantitativ-diskrete Mutationsindikatoren
    quant_discrete_mutation_indicators = [
        'DNMT3A', 'PTPN11', 'WT1', 'ASXL1', 'ATRX', 'BCOR', 'BCORL1', 'BRAF',
        'CALR', 'CBL', 'CBLB', 'CDKN2A', 'CEBPA.bZIP', 'CEBPA.NGS', 'CEBPA.TAD1',
        'CEBPA.TAD2', 'CSF3R', 'CUX1', 'dmCEBPA', 'ETV6', 'EZH2', 'FBXW7',
        'FLT3.ITD', 'FLT3.TKD', 'GATA1', 'GATA2', 'GNAS', 'HRAS', 'IKZF1',
        'KDM6A', 'KIT', 'KRAS', 'MPL', 'MYD88', 'NOTCH1', 'NPM1.NGS', 'NRAS',
        'PDGFRA', 'PHF6', 'PTEN', 'RAD21', 'RUNX1', 'SETBP1', 'SF3B1', 'SFRS2',
        'SMC1A', 'SMC3', 'smCEBPA', 'STAG2', 'TET2', 'TP53', 'U2AF1', 'ZRSR2',
        "t(8;21)", "inv(16) or t(16;16)", "t(9;11)", "t(6;9)", "t(9;22)", "inv(3) or t(3;3)", "del5/5q", "del7", "del17"]

    #: Irrelevante Attribute
    irrelevant = [
        'ALSCTCR1', 'ALSCTDTC', 'ALSCTSLV', 'DoubleInduction', 'IT1RES',
        'PatMatID', 'PTF', 'PTFDTC', 'R1', 'R1DTC', 'R2', 'R2DTC', 'R3',
        'R3DTC', 'TIDTC', 'TRIALID', 'TRT', 'TRTIND', 'CD34U']

    #: Datumsangaben
    dates = [
        'CR1DTC', 'DTHDTC', 'EFSDTC', 'IT1STDTC', 'OSDTC', 'RFSDTC']

    #: :ref:`Textuelle Attribute`
    text = [
        'CGKT', 'DNMT3AT']
