SELECT
    amedas_kobe.latitude_velocity,
    amedas_kobe.longitude_velocity,
    amedas_kobe.temperature,
    nowphas_kobe.significant_height,
    nowphas_kobe.significant_period
FROM 
    amedas_kobe
    INNER JOIN nowphas_kobe ON amedas_kobe.datetime = nowphas_kobe.datetime 
    JOIN purpose01 ON amedas_kobe.datetime = purpose01.datetime
WHERE
    amedas_kobe.inferiority = 0 AND
    nowphas_kobe.inferiority = 0 AND
    purpose01.purpose = 'eval'
;