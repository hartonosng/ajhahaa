WITH features AS (
    SELECT 
        customer_id,
        partition_date,
        aum AS L0M_aum,
        LAG(aum, 1) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L1M_aum,
        LAG(aum, 2) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L2M_aum,
        LAG(aum, 3) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L3M_aum,
        LAG(aum, 4) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L4M_aum,
        LAG(aum, 5) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L5M_aum,
        LAG(aum, 6) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L6M_aum,
        casa_balance AS L0M_casa_balance,
        LAG(casa_balance, 1) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L1M_casa_balance,
        LAG(casa_balance, 2) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L2M_casa_balance,
        LAG(casa_balance, 3) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L3M_casa_balance,
        LAG(casa_balance, 4) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L4M_casa_balance,
        LAG(casa_balance, 5) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L5M_casa_balance,
        LAG(casa_balance, 6) OVER (PARTITION BY customer_id ORDER BY partition_date) AS L6M_casa_balance
    FROM 
        your_table
    WHERE 
        partition_date BETWEEN '20230131' AND '20231231'
)
