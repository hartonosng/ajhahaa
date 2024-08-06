WITH json_parts AS (
    SELECT
        customer_id,
        product_type,
        CONCAT('{"account_id":"', account_id, '","current_balance":', current_balance, '}') AS account_json
    FROM accounts
),

json_array AS (
    SELECT
        customer_id,
        product_type,
        CONCAT('[', GROUP_CONCAT(account_json, ','), ']') AS accounts_json
    FROM json_parts
    GROUP BY customer_id, product_type
),

json_object AS (
    SELECT
        customer_id,
        CONCAT('{"', product_type, '":', accounts_json, '}') AS product_json
    FROM json_array
),

final_json AS (
    SELECT
        customer_id,
        CONCAT('{', GROUP_CONCAT(product_json, ','), '}') AS account_json
    FROM json_object
    GROUP BY customer_id
)

SELECT customer_id, account_json FROM final_json;
