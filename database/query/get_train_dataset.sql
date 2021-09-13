SELECT
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
    amedas_kobe
    INNER JOIN amedas_kix ON amedas_kobe.datetime = amedas_kix.datetime 
    INNER JOIN amedas_tomogashima ON amedas_kobe.datetime = amedas_tomogashima.datetime 
    INNER JOIN nowphas_kobe ON amedas_kobe.datetime = nowphas_kobe.datetime 
    JOIN purpose ON amedas_kobe.datetime = purpose.datetime
WHERE
    amedas_kobe.inferiority = 0 AND
    amedas_kix.inferiority = 0 AND
    amedas_tomogashima.inferiority = 0 AND
    nowphas_kobe.inferiority = 0 AND
    purpose.purpose = 'train'
order by amedas_kobe.datetime
;