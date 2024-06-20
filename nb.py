WITH filtered_data AS (
    SELECT 
        customer_id, 
        partition_date, 
        collectibility
    FROM 
        your_table
    WHERE 
        partition_date BETWEEN '20230131' AND '20240331'  -- Include 3 months after the end of 2023
),
collectibility_data AS (
    SELECT 
        customer_id, 
        partition_date, 
        collectibility,
        LEAD(collectibility, 1) OVER (PARTITION BY customer_id ORDER BY partition_date) AS collectibility_next_1,
        LEAD(collectibility, 2) OVER (PARTITION BY customer_id ORDER BY partition_date) AS collectibility_next_2,
        LEAD(collectibility, 3) OVER (PARTITION BY customer_id ORDER BY partition_date) AS collectibility_next_3
    FROM 
        filtered_data
)
SELECT 
    customer_id,
    partition_date,
    collectibility,
    CASE 
        WHEN collectibility = 1 AND (
            (collectibility_next_1 IS NOT NULL AND collectibility_next_1 >= 2) OR
            (collectibility_next_2 IS NOT NULL AND collectibility_next_2 >= 2) OR
            (collectibility_next_3 IS NOT NULL AND collectibility_next_3 >= 2)
        ) THEN 1 
        ELSE 0 
    END AS label
FROM 
    collectibility_data
WHERE 
    partition_date BETWEEN '20230131' AND '20231231'
ORDER BY 
    customer_id, 
    partition_date;
