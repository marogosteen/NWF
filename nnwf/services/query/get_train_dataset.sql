SELECT
    amedas_kobe.datetime,
    amedas_kobe.inferiority == 1 OR 
    amedas_tomogashima.inferiority == 1 OR 
    amedas_kix.inferiority == 1 OR
    nowphas_kobe.inferiority == 1 AS
        inferiority,
    amedas_kobe.latitude_velocity,
    amedas_kobe.longitude_velocity,
    amedas_kobe.temperature,
    amedas_kix.latitude_velocity,
    amedas_kix.longitude_velocity,
    amedas_kix.temperature,
    amedas_tomogashima.latitude_velocity,
    amedas_tomogashima.longitude_velocity,
    amedas_tomogashima.temperature,
    nowphas_kobe.significant_height,
    nowphas_kobe.significant_period
FROM
    amedas AS amedas_kobe
    INNER JOIN amedas AS amedas_kix ON amedas_kobe.datetime == amedas_kix.datetime
    INNER JOIN amedas AS amedas_tomogashima ON amedas_kobe.datetime == amedas_tomogashima.datetime
    INNER JOIN nowphas AS nowphas_kobe ON amedas_kobe.datetime == nowphas_kobe.datetime
WHERE
    amedas_kobe.place == 'kobe' AND
    amedas_kix.place == 'kix' AND
    amedas_tomogashima.place == 'tomogashima' AND
    amedas_kobe.datetime >= 201601010000 AND
    amedas_kobe.datetime <= 201812312300
ORDER BY
    amedas_kobe.datetime
;