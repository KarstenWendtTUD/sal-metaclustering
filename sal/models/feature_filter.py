#region imports

#endregion

#region base class
class SALA_FeatureFilter:
    name = ""

    def __init__(self, name = "not set"):
        self.name = name

    def run(self):
        pass
#endregion

#region filters
class SALA_ListFilter(SALA_FeatureFilter):
    def run(self, df):
        droppable = [
            'ZRSR', 'WT', 'U2AF', 'TP', 'TET', 'STAG', 'SMC', 'SFRS', 'SETBP', 'SF', 'RUNX', 'RAD', 'PTPN', 'PTEN',
            'PHF', 'PDGFRA', 'NRAS', 'NPM', 'NOTCH', 'MYD', 'MPL', 'KRAS', 'KIT', 'KDM', 'JAK', 'IKFZ', 'IDH', 'IKZF',
            'HRAS', 'GNAS', 'GATA', 'FLT', 'FBXW', 'EZH', 'ETV', 'DNMT', 'CUX', 'CSF', 'CEBPA', 'CDKN', 'CBL', 'CALR',
            'BRAF', 'BCOR', 'ATRX', 'ASXL',  # wegen Fehlen in anderer Studie?
            'OSSTAT', 'RFSSTAT', 'EFSSTAT', 'ED60', 'ED30',
            'CISTAT', 'CR1DTC', 'RFSDTC', 'OSDTC', 'DTHDTC', 'EFSDTC',
            'ELNRisk', 'CGELN', 'CGSTUD', 'ECOG', 'FAB', 'WHO',
            'D16BMB',  # angenommen wir sind nicht schon am Tag 16 vorbeigekommen
            'IT1STDTC'  # ungeachtet des absoluten Beginns der Induktionsther.
        ]

        targets = []
        for feature in df.columns:
            for drop in droppable:
                if str(feature).startswith(drop):
                    targets.append(str(feature))

        df = df.drop(columns=targets)

        return df
#endregion