SELECT
    COUNT(amedas_kobe.datetime)
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
;