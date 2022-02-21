
class DbQuery():
    def __init__(self, targetyear: int, mode: str, querySavePath: str = None) -> None:
        """
        fetchする期間を指定し，sql queryを返す．

        Args:
        -----
            - targetyear (int): 予測対象年
            - mode (str): modeは"train"または"eval"．
            "tarain"の場合はtargetyear以外の期間からrecordをfetchする．
            "eval"の場合はtargetyearの年からrecordをfetchする．
            - querySavePath (str)
        """

        if mode == "train":
            yes_or_no = "NOT"
        elif mode == "eval":
            yes_or_no = ""
        else:
            errmse = f'modeとして {mode} は受け付けません．modeは"train"または"eval"で指定してください．'
            exit(errmse)

        self.__generateQuery(targetyear, yes_or_no)

        if querySavePath:
            with open(querySavePath, mode="w") as f:
                f.write(self.query)

    def __generateQuery(self, targetyear: int, yes_or_no: str):
        self.query = f"""
SELECT
    ukb.datetime,

    ukb.velocity,
    ukb.sin_direction,
    ukb.cos_direction,
    kix.velocity,
    kix.sin_direction,
    kix.cos_direction,
    tomogashima.velocity,
    tomogashima.sin_direction,
    tomogashima.cos_direction,
    akashi.velocity,
    akashi.sin_direction,
    akashi.cos_direction,
    osaka.velocity,
    osaka.sin_direction,
    osaka.cos_direction,

    Temperature.temperature,

    fukuiPressure.air_pressure,
    fukuyamaPressure.air_pressure,
    hamadaPressure.air_pressure,
    hikonePressure.air_pressure,
    himejiPressure.air_pressure,
    hiroshimaPressure.air_pressure,
    kobePressure.air_pressure,
    kochiPressure.air_pressure,
    kurePressure.air_pressure,
    kyotoPressure.air_pressure,
    maizuruPressure.air_pressure,
    matsuePressure.air_pressure,
    matsuyamaPressure.air_pressure,
    murotomisakiPressure.air_pressure,
    naraPressure.air_pressure,
    okayamaPressure.air_pressure,
    osakaPressure.air_pressure,
    owasePressure.air_pressure,
    saigoPressure.air_pressure,
    sakaiPressure.air_pressure,
    shimizuPressure.air_pressure,
    shionomisakiPressure.air_pressure,
    sukumoPressure.air_pressure,
    sumotoPressure.air_pressure,
    tadotsuPressure.air_pressure,
    takamatsuPressure.air_pressure,
    tokushimaPressure.air_pressure,
    tottoriPressure.air_pressure,
    toyookaPressure.air_pressure,
    tsuPressure.air_pressure,
    tsurugaPressure.air_pressure,
    tsuyamaPressure.air_pressure,
    uenoPressure.air_pressure,
    uwajimaPressure.air_pressure,
    wakayamaPressure.air_pressure,
    yokkaichiPressure.air_pressure,
    yonagoPressure.air_pressure,

    Wave.significant_height,
    Wave.significant_period

FROM
    Wind AS ukb
    INNER JOIN Wind AS kix ON ukb.datetime == kix.datetime
    INNER JOIN Wind AS tomogashima ON ukb.datetime == tomogashima.datetime
    INNER JOIN Wind AS akashi ON ukb.datetime == akashi.datetime
    INNER JOIN Wind AS osaka ON ukb.datetime == osaka.datetime
    INNER JOIN Temperature ON ukb.datetime == Temperature.datetime

    INNER JOIN AirPressure AS fukuiPressure ON ukb.datetime == fukuiPressure.datetime
    INNER JOIN AirPressure AS fukuyamaPressure ON ukb.datetime == fukuyamaPressure.datetime
    INNER JOIN AirPressure AS hamadaPressure ON ukb.datetime == hamadaPressure.datetime
    INNER JOIN AirPressure AS hikonePressure ON ukb.datetime == hikonePressure.datetime
    INNER JOIN AirPressure AS himejiPressure ON ukb.datetime == himejiPressure.datetime
    INNER JOIN AirPressure AS hiroshimaPressure ON ukb.datetime == hiroshimaPressure.datetime
    INNER JOIN AirPressure AS kobePressure ON ukb.datetime == kobePressure.datetime
    INNER JOIN AirPressure AS kochiPressure ON ukb.datetime == kochiPressure.datetime
    INNER JOIN AirPressure AS kurePressure ON ukb.datetime == kurePressure.datetime
    INNER JOIN AirPressure AS kyotoPressure ON ukb.datetime == kyotoPressure.datetime
    INNER JOIN AirPressure AS maizuruPressure ON ukb.datetime == maizuruPressure.datetime
    INNER JOIN AirPressure AS matsuePressure ON ukb.datetime == matsuePressure.datetime
    INNER JOIN AirPressure AS matsuyamaPressure ON ukb.datetime == matsuyamaPressure.datetime
    INNER JOIN AirPressure AS murotomisakiPressure ON ukb.datetime == murotomisakiPressure.datetime
    INNER JOIN AirPressure AS naraPressure ON ukb.datetime == naraPressure.datetime
    INNER JOIN AirPressure AS okayamaPressure ON ukb.datetime == okayamaPressure.datetime
    INNER JOIN AirPressure AS osakaPressure ON ukb.datetime == osakaPressure.datetime
    INNER JOIN AirPressure AS owasePressure ON ukb.datetime == owasePressure.datetime
    INNER JOIN AirPressure AS saigoPressure ON ukb.datetime == saigoPressure.datetime
    INNER JOIN AirPressure AS sakaiPressure ON ukb.datetime == sakaiPressure.datetime
    INNER JOIN AirPressure AS shimizuPressure ON ukb.datetime == shimizuPressure.datetime
    INNER JOIN AirPressure AS shionomisakiPressure ON ukb.datetime == shionomisakiPressure.datetime
    INNER JOIN AirPressure AS sukumoPressure ON ukb.datetime == sukumoPressure.datetime
    INNER JOIN AirPressure AS sumotoPressure ON ukb.datetime == sumotoPressure.datetime
    INNER JOIN AirPressure AS tadotsuPressure ON ukb.datetime == tadotsuPressure.datetime
    INNER JOIN AirPressure AS takamatsuPressure ON ukb.datetime == takamatsuPressure.datetime
    INNER JOIN AirPressure AS tokushimaPressure ON ukb.datetime == tokushimaPressure.datetime
    INNER JOIN AirPressure AS tottoriPressure ON ukb.datetime == tottoriPressure.datetime
    INNER JOIN AirPressure AS toyookaPressure ON ukb.datetime == toyookaPressure.datetime
    INNER JOIN AirPressure AS tsuPressure ON ukb.datetime == tsuPressure.datetime
    INNER JOIN AirPressure AS tsurugaPressure ON ukb.datetime == tsurugaPressure.datetime
    INNER JOIN AirPressure AS tsuyamaPressure ON ukb.datetime == tsuyamaPressure.datetime
    INNER JOIN AirPressure AS uenoPressure ON ukb.datetime == uenoPressure.datetime
    INNER JOIN AirPressure AS uwajimaPressure ON ukb.datetime == uwajimaPressure.datetime
    INNER JOIN AirPressure AS wakayamaPressure ON ukb.datetime == wakayamaPressure.datetime
    INNER JOIN AirPressure AS yokkaichiPressure ON ukb.datetime == yokkaichiPressure.datetime
    INNER JOIN AirPressure AS yonagoPressure ON ukb.datetime == yonagoPressure.datetime

    INNER JOIN Wave ON ukb.datetime == Wave.datetime

WHERE
    ukb.place == 'ukb' AND
    kix.place == 'kix' AND
    tomogashima.place == 'tomogashima' AND
    akashi.place == 'akashi' AND
    osaka.place == 'osaka' AND

    fukuiPressure.place == 'fukui' AND
    fukuyamaPressure.place == 'fukuyama' AND
    hamadaPressure.place == 'hamada' AND
    hikonePressure.place == 'hikone' AND
    himejiPressure.place == 'himeji' AND
    hiroshimaPressure.place == 'hiroshima' AND
    kobePressure.place == 'kobe' AND
    kochiPressure.place == 'kochi' AND
    kurePressure.place == 'kure' AND
    kyotoPressure.place == 'kyoto' AND
    maizuruPressure.place == 'maizuru' AND
    matsuePressure.place == 'matsue' AND
    matsuyamaPressure.place == 'matsuyama' AND
    murotomisakiPressure.place == 'murotomisaki' AND
    naraPressure.place == 'nara' AND
    okayamaPressure.place == 'okayama' AND
    osakaPressure.place == 'osaka' AND
    owasePressure.place == 'owase' AND
    saigoPressure.place == 'saigo' AND
    sakaiPressure.place == 'sakai' AND
    shimizuPressure.place == 'shimizu' AND
    shionomisakiPressure.place == 'shionomisaki' AND
    sukumoPressure.place == 'sukumo' AND
    sumotoPressure.place == 'sumoto' AND
    tadotsuPressure.place == 'tadotsu' AND
    takamatsuPressure.place == 'takamatsu' AND
    tokushimaPressure.place == 'tokushima' AND
    tottoriPressure.place == 'tottori' AND
    toyookaPressure.place == 'toyooka' AND
    tsuPressure.place == 'tsu' AND
    tsurugaPressure.place == 'tsuruga' AND
    tsuyamaPressure.place == 'tsuyama' AND
    uenoPressure.place == 'ueno' AND
    uwajimaPressure.place == 'uwajima' AND
    wakayamaPressure.place == 'wakayama' AND
    yokkaichiPressure.place == 'yokkaichi' AND
    yonagoPressure.place == 'yonago' AND

    {yes_or_no}(
        datetime(ukb.datetime) >= datetime("{targetyear}-01-01 00:00") AND
        datetime(ukb.datetime) <= datetime("{targetyear}-12-31 23:00")
    )

ORDER BY
    ukb.datetime
;
"""
